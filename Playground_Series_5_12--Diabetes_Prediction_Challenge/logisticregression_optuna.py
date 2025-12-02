import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from competition_specifics import COMPETITION_NAME, load_and_prepare, N_FOLDS, TARGET_COL
sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_api_functions import submit_prediction


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)


def target_encode_with_original_data(df: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
    for col in [c for c in df.columns if c != TARGET_COL]:
        te_col = f"TEO_{col}"
        tmp_df = orig.groupby(col, observed=True)[TARGET_COL].mean()
        tmp_df.name = te_col
        df = df.merge(tmp_df, on=col, how='left')
        df[te_col] = df[te_col].fillna(orig[TARGET_COL].mean())
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """ Prepare categorical and numerical columns.
    Args:
        df (pd.DataFrame): Input data.
    Returns:
        ColumnTransformer: Preprocessor for the data.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist()
                      if c != TARGET_COL]

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
        ])

    return preprocessor


def train_logreg_optuna(df: pd.DataFrame, test_df: pd.DataFrame
                        ) -> tuple[np.ndarray, list[np.ndarray], float]:
    """ Run the optuna study, train the final model with the best parameters and generate
        predictions.
    Args:
        df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
    Returns:
        tuple[np.ndarray, list[np.ndarray], float]: Out-of-fold predictions, test predictions,
            score of the best trial
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values

    preprocessor = build_preprocessor(df)

    def objective(trial: optuna.trial.Trial) -> float:
        C = trial.suggest_float('C', 1e-3, 10, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg'])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])

        model = LogisticRegression(C=C,
                                   solver=solver,
                                   max_iter=1000,
                                   class_weight=class_weight,
                                   n_jobs=-1)

        pipe = Pipeline(steps=[('preprocess', preprocessor),
                               ('log_reg_class', model)])

        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pipe.fit(X_train, y_train)
            preds = pipe.predict_proba(X_val)[:, 1]

            fold_auc = roc_auc_score(y_val, preds)
            scores.append(fold_auc)

            trial.report(fold_auc, step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(direction='maximize',
                                study_name='logreg_optimization',
                                storage='sqlite:///optuna_study.db',
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=1000, timeout=60*60*6)


    # Train final model using best hyperparameters
    best_params = study.best_params

    final_model = LogisticRegression(C=best_params['C'],
                                     solver=best_params['solver'],
                                     class_weight=best_params['class_weight'],
                                     max_iter=1000,
                                     n_jobs=-1)

    final_pipe = Pipeline(steps=[('preprocess', preprocessor),
                                 ('log_reg_class', final_model)])

    # OOF evaluation
    oof_preds = np.zeros(len(df))
    test_fold_preds = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, _ = y[train_idx], y[val_idx]

        final_pipe.fit(X_train, y_train)
        oof_preds[val_idx] = final_pipe.predict_proba(X_val)[:, 1]

        test_pred = final_pipe.predict_proba(test_df)[:, 1]
        test_fold_preds.append(test_pred)

    return oof_preds, test_fold_preds, study.best_value


X_train = load_and_prepare('train.csv')
X_test = load_and_prepare('test.csv')
orig = load_and_prepare('original.csv')

#X_train = target_encode_with_original_data(X_train, orig)
#X_test = target_encode_with_original_data(X_test, orig)

#X_train = pd.concat([X_train, orig], ignore_index=True)

oof_preds, test_fold_preds, best_value = train_logreg_optuna(X_train, X_test)

# write files for ensembling
OOF_DF = pd.DataFrame({'y_true': X_train[TARGET_COL], 'oof': oof_preds})
OOF_DF.to_csv('oof.csv', index=False)

submit_df = pd.read_csv('sample_submission.csv')
submit_df[TARGET_COL] = np.mean(np.array(test_fold_preds), axis=0).astype(float)
submit_df.to_csv('predictions_optuna.csv', columns=['id', TARGET_COL], index=False)

# submit to kaggle
public_score = submit_prediction(COMPETITION_NAME, 'predictions_optuna.csv',
                                 f"LogisticRegression optuna ({best_value})")
print(f'Public score: {public_score}')

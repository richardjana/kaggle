import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from competition_specifics import COMPETITION_NAME, load_and_prepare, N_FOLDS, TARGET_COL
sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_api_functions import submit_prediction


def encode_tree(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """ Ordinal encoding for tree-based models. df is modified in-place and returned for
        convenience.
    Args:
        df (pd.DataFrame): DataFrame to encode.
        cat_cols (list[str]): List of columns to encode.
    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    for col in cat_cols:
        df[col] = df[col].cat.codes

    return df

def target_encode_with_original_data(df: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
    for col in [c for c in df.columns if c != TARGET_COL]:
        te_col = f"TEO_{col}"
        tmp_df = orig.groupby(col, observed=True)[TARGET_COL].mean()
        tmp_df.name = te_col
        df = df.merge(tmp_df, on=col, how='left')
        df[te_col] = df[te_col].fillna(orig[TARGET_COL].mean())
    return df


# Load dataset
X_train = load_and_prepare('train.csv')
X_test = load_and_prepare('test.csv')
orig = load_and_prepare('original.csv')

CAT_COLS = [c for c in X_train.select_dtypes(include=['category']).columns if c != TARGET_COL]
NUM_COLS = [c for c in X_train.select_dtypes(include=['number']).columns if c != TARGET_COL]

#X_train = target_encode_with_original_data(X_train, orig)
#X_test = target_encode_with_original_data(X_test, orig)

#X_train = pd.concat([X_train, orig], ignore_index=True)

X_train = encode_tree(X_train, CAT_COLS)
X_test = encode_tree(X_test, CAT_COLS)
orig = encode_tree(orig, CAT_COLS)


ADDITIONAL_PARAMS = {'n_jobs': -1,
                     'random_state': 77,
                     'class_weight': 'balanced'}


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Optuna objective
def objective(trial: optuna.trial.Trial) -> float:
    params = {'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
              'max_depth': trial.suggest_int('max_depth', 3, 40),
              'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
              'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
              'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
              'bootstrap': trial.suggest_categorical('bootstrap', [True, False])}
    params.update(ADDITIONAL_PARAMS)

    aucs = []
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, X_train[TARGET_COL])):
        X_train_fold = X_train.iloc[train_idx].copy()
        X_valid_fold = X_train.iloc[valid_idx].copy()

        y_train_fold = X_train_fold.pop(TARGET_COL)
        y_valid_fold = X_valid_fold.pop(TARGET_COL)

        model = RandomForestClassifier(**params)
        model.fit(X_train_fold, y_train_fold)

        y_pred = model.predict_proba(X_valid_fold)[:, 1]
        fold_auc = roc_auc_score(y_valid_fold, y_pred)
        aucs.append(fold_auc)

        trial.report(fold_auc, step=fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(aucs)


# Create and optimize Optuna study
study = optuna.create_study(direction='maximize',
                            study_name='diabetes',
                            storage='sqlite:///optuna_study.db',
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
study.optimize(objective, n_trials=10_000, timeout=60*60*6)


# Train final model with best parameters
best_params = study.best_params
best_params.update(ADDITIONAL_PARAMS)

oof_preds = np.zeros(len(X_train))
test_fold_preds = []

for train_idx, valid_idx in skf.split(X_train, X_train[TARGET_COL]):
    X_train_fold = X_train.iloc[train_idx].copy()
    X_valid_fold = X_train.iloc[valid_idx].copy()

    y_train_fold = X_train_fold.pop(TARGET_COL)
    y_valid_fold = X_valid_fold.pop(TARGET_COL)

    model = RandomForestClassifier(**best_params)
    model.fit(X_train_fold, y_train_fold)

    oof_preds[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
    test_fold_preds.append(model.predict_proba(X_test)[:, 1])


# write files for ensembling
OOF_DF = pd.DataFrame({'y_true': X_train[TARGET_COL], 'oof': oof_preds})
OOF_DF.to_csv('oof.csv', index=False)

submit_df = pd.read_csv('sample_submission.csv')
submit_df[TARGET_COL] = np.mean(np.array(test_fold_preds), axis=0).astype(float)
submit_df.to_csv('predictions_optuna.csv', columns=['id', TARGET_COL], index=False)

# submit to kaggle
public_score = submit_prediction(COMPETITION_NAME, 'predictions_optuna.csv',
                                 f"RF optuna ({study.best_value})")
print(f'Public score: {public_score}')

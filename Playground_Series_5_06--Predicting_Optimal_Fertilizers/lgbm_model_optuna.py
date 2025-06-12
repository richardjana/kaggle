from typing import Tuple
import sys

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import mapk  # noqa
from kaggle_api_functions import submit_prediction
from competition_specifics import TARGET_COL, COMPETITION_NAME, load_preprocess_data


def lgb_map3_eval(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    """ Custom LightGBM eval function to compute MAP@3.
    Parameters:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Array of predictions.
    Returns:
        Tuple[str, float, bool]: metric name, value, higher_is_better
    """
    top_3 = np.argsort(-y_pred, axis=1)[:, :3]  # Compute top 3 predictions
    actual = [[label] for label in y_true]  # Format true labels for mapk

    return 'map@3', mapk(actual, top_3.tolist(), k=3), True


def make_prediction(model: lgb.LGBMClassifier, test_df: pd.DataFrame) -> None:
    """ Make a prediction for the test data, with a given model.
    Args:
        model (lgb.LGBMRegressor): Model used for the prediction.
        test_df (pd.DataFrame): DataFrame with the test data, pre-processed.
    """
    submit_df = pd.read_csv('sample_submission.csv')
    pred_proba = np.asarray(model.predict_proba(test_df.to_numpy()))
    top_3_indices = np.argsort(-pred_proba, axis=1)[:, :3]

    top_3_labels = np.array([
        encoders[TARGET_COL].inverse_transform(sample_top3)
        for sample_top3 in top_3_indices
    ])

    joined_top3 = [' '.join(labels) for labels in top_3_labels]

    submit_df[TARGET_COL] = joined_top3
    submit_df.to_csv('predictions_LGBM_optuna.csv',
                     columns=['id', TARGET_COL], index=False)


# Load dataset
train, test, encoders = load_preprocess_data('LGBM')

best_iterations = []

# Define objective function for Optuna
def objective(trial):
    best_iteration_folds = []

    param = {
        'objective': 'multiclass',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mapa3_scores = []

    for train_idx, val_idx in kf.split(train):
        df_train_fold, df_val_fold = train.iloc[train_idx].copy(), train.iloc[val_idx].copy()

        te = TargetEncoder(target_type='multiclass', cv=5, shuffle=True, random_state=42)
        preprocessor = ColumnTransformer(transformers=[('te', te, ['sc-interaction'])],
                                         remainder='passthrough',
                                         verbose_feature_names_out=False)
        preprocessor.set_output(transform='pandas')

        X_train_fold = preprocessor.fit_transform(df_train_fold, df_train_fold[TARGET_COL])
        X_val_fold = preprocessor.transform(df_val_fold)

        y_train_fold = df_train_fold.pop(TARGET_COL)
        y_val_fold = df_val_fold.pop(TARGET_COL)

        model = lgb.LGBMClassifier(**param)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.callback.early_stopping(stopping_rounds=100),
                             lgb.callback.log_evaluation(period=0)
                             ]
                 )

        pred_val = np.asarray(model.predict_proba(X_val_fold))

        top_3 = np.argsort(-pred_val, axis=1)[:, :3]
        actual = [[int(label)] for label in y_val_fold]
        mapa3_scores.append(mapk(actual, top_3.tolist(), k=3))

        best_iteration_folds.append(model.best_iteration_)

    best_iterations.append(int(np.mean(best_iteration_folds)))

    return np.mean(mapa3_scores)

# Create and optimize Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10_000, timeout=60*60*11)

best_params = study.best_params
best_params.pop('n_estimators', best_iterations[study.best_trial.number])

# Print best trial
print("Best trial:")
print(f"  MAP@3: {study.best_value}")
print("  Best hyperparameters:")
for key, value in best_params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
te = TargetEncoder(target_type='multiclass', cv=5, shuffle=True, random_state=42)
preprocessor = ColumnTransformer(transformers=[('te', te, ['sc-interaction'])],
                                 remainder='passthrough',
                                 verbose_feature_names_out=False)
preprocessor.set_output(transform='pandas')

y = train.pop(TARGET_COL)
train = preprocessor.fit_transform(train, y)
test = preprocessor.transform(test)

model = lgb.LGBMClassifier(**best_params)
model.fit(train, y)
joblib.dump(model, 'lgb_model.pkl')

# make prediction for the test data
make_prediction(model, test)

public_score = submit_prediction(COMPETITION_NAME, 'predictions_LGBM_optuna.csv',
                                 f"LGBM optuna ({study.best_value})")
print(f'Public score: {public_score}')

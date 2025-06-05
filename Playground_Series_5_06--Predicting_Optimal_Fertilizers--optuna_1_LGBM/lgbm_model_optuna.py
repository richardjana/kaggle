from typing import Dict, List, Tuple
import sys

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import mapk  # noqa
from kaggle_api_functions import submit_prediction
from prepare_data import load_preprocess_data

TARGET_COL = 'Fertilizer Name'


def lgb_map3_eval(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    """
    Custom LightGBM eval function to compute MAP@3.
    Parameters:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Flat array of predictions (num_samples * num_classes).
    Returns:
        Tuple[str, float, bool]: metric name, value, higher_is_better
    """
    # Compute top 3 predictions
    top_3 = np.argsort(-y_pred, axis=1)[:, :3]

    # Format true labels for mapk
    actual = [[label] for label in y_true]

    # Compute MAP@3
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
dataframe, test, encoders = load_preprocess_data('train.csv', 'test.csv', TARGET_COL, 'LGBM')

# Split into train/test sets
y = dataframe.pop(TARGET_COL).to_numpy()
X = dataframe.to_numpy()
categorical_features = [dataframe.columns.get_loc(c) for c in ['Soil Type', 'Crop Type']]

# Define objective function for Optuna
def objective(trial):
    param = {
        'objective': 'multiclass',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 5000,
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mapa3_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=lgb_map3_eval,
                  callbacks=[lgb.callback.early_stopping(stopping_rounds=100),
                             lgb.callback.log_evaluation(period=0)
            ]
        )

        pred_val = np.asarray(model.predict_proba(X_val))
        #pred_val_labels = encoders[TARGET_COL].inverse_transform(pred_val.argmax(axis=1))

        top_3 = np.argsort(-pred_val, axis=1)[:, :3]
        actual = [[int(label)] for label in y_val]
        mapa3_scores.append(mapk(actual, top_3.tolist(), k=3))

    return np.mean(mapa3_scores)

# Create and optimize Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000, timeout=60*60*11)

# Print best trial
print("Best trial:")
print(f"  MAP@3: {study.best_value}")
print("  Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
best_params = study.best_params

model = lgb.LGBMClassifier(**best_params)

eval_results = {}

model.fit(
    X, y,
    eval_set=[(X, y)],
    eval_metric=lgb_map3_eval,
    callbacks=[
        lgb.callback.early_stopping(stopping_rounds=100),
        lgb.callback.log_evaluation(period=50),
        lgb.callback.record_evaluation(eval_results)
    ]
)

joblib.dump(model, 'lgb_model.pkl')

# Evaluate
pred_train = np.asarray(model.predict_proba(X))
top_3 = np.argsort(-pred_train, axis=1)[:, :3]
actual = [[int(label)] for label in y]

map3_score = mapk(actual, top_3.tolist(), k=3)
print(f'MAP@3 score: {map3_score:.7f}')

# make prediction for the test data
make_prediction(model, test)

public_score = submit_prediction('playground-series-s5e6', 'predictions_LGBM_optuna.csv', f"LGBM optuna ({map3_score})")
print(f'Public score: {public_score:.7f}')

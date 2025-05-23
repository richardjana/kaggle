from typing import Dict, List

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_log_error

import sys
sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import make_diagonal_plot, rmsle  # noqa
from prepare_calories_data import load_preprocess_data

TARGET_COL = 'Calories'


# Load dataset
log1p_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
dataframe, test = load_preprocess_data('train.csv', 'test.csv', TARGET_COL, log1p_transformer)

# Split into train/test sets
y = dataframe.pop(TARGET_COL).to_numpy()
X = dataframe.to_numpy()

# Define objective function for Optuna
def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
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
    rmsle_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', 
                  callbacks=[lgb.callback.early_stopping(stopping_rounds=100),
                             lgb.callback.log_evaluation(period=0)
            ]
        )

        preds = model.predict(X_val)
        preds = log1p_transformer.inverse_transform(preds.reshape(-1, 1))
        y_val_trans = log1p_transformer.inverse_transform(y_val.reshape(-1, 1))

        rmsle_score = rmsle(y_val_trans, preds)
        rmsle_scores.append(rmsle_score)

    return np.mean(rmsle_scores)

# Create and optimize Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000, timeout=75600)  # Adjust timeout or n_trials as needed

# Print best trial
print("Best trial:")
print(f"  RMSLE: {study.best_value}")
print("  Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'n_estimators': 5000
})

model = lgb.LGBMRegressor(**best_params)

eval_results = {}

model.fit(
    X, y,
    eval_set=[(X, y)],
    eval_metric='rmse',
    callbacks=[
        lgb.callback.early_stopping(stopping_rounds=100),
        lgb.callback.log_evaluation(period=50),
        lgb.callback.record_evaluation(eval_results)
    ]
)

joblib.dump(model, 'lgb_model.pkl')

# Predict
pred_train = model.predict(X)
pred_train = log1p_transformer.inverse_transform(pred_train.reshape(-1, 1))

# Evaluate
rmsle_final = rmsle(y, pred_train)
print(f'Root Mean Squared Logarithmic Error: {rmsle_final:.7f}')

# Make prediction for the test data
test_preds = model.predict(test)
test_preds = log1p_transformer.inverse_transform(test_preds.reshape(-1, 1))

submit_df = pd.read_csv('sample_submission.csv')
submit_df[TARGET_COL] = test_preds
submit_df.to_csv(f"predictions_LGBM.csv", columns=['id', TARGET_COL], index=False)

y = log1p_transformer.inverse_transform(y)

make_diagonal_plot(pd.DataFrame({TARGET_COL: y.ravel(), 'PREDICTION': pred_train.ravel()}),
                        pd.DataFrame({TARGET_COL: y.ravel(), 'PREDICTION': pred_train.ravel()}),
                        TARGET_COL, rmsle, 'RMSLE',
                        f"error_diagonal_LGBM_optuna.png", precision=5)

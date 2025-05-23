from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
from xgboost.sklearn import XGBRegressor

import sys
sys.path.append('../')
from kaggle_utilities import make_diagonal_plot, rmsle # noqa
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
        'objective': 'reg:squarederror',
        'n_estimators': 10000,  # use early stopping
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'tree_method': 'hist',  # use 'hist' for faster training
        'verbosity': 0,
        'early_stopping_rounds': 100,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmsle_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBRegressor(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
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

model = XGBRegressor(**best_params)
model.fit(X, y, eval_set=[(X, y)], verbose=50)
eval_results = model.evals_result()

joblib.dump(model, 'xgb_model.pkl')

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
                        f"error_diagonal_XGB_optuna.png", precision=5)

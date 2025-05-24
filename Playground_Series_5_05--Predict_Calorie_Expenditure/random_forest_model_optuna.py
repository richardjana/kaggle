import joblib
import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append('/'.join(__file__.split('/')[:-2]))
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
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmsle_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = RandomForestRegressor(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        preds = log1p_transformer.inverse_transform(preds.reshape(-1, 1))
        y_val_trans = log1p_transformer.inverse_transform(y_val.reshape(-1, 1))

        rmsle_score = rmsle(y_val_trans, preds)
        rmsle_scores.append(rmsle_score)

    return np.mean(rmsle_scores)

# Create and optimize Optuna study
study = optuna.create_study(direction='minimize', sampler=TPESampler())
study.optimize(objective, n_trials=1000, timeout=75600)  # Adjust timeout or n_trials as needed

# Print best trial
print("Best trial:")
print(f"  RMSLE: {study.best_value}")
print("  Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
best_params = study.best_params

model = RandomForestRegressor(**best_params, n_jobs=-1)
model.fit(X, y)

joblib.dump(model, 'rf_model.pkl')

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
submit_df.to_csv(f"predictions_RF.csv", columns=['id', TARGET_COL], index=False)

y = log1p_transformer.inverse_transform(y)

make_diagonal_plot(pd.DataFrame({TARGET_COL: y.ravel(), 'PREDICTION': pred_train.ravel()}),
                        pd.DataFrame({TARGET_COL: y.ravel(), 'PREDICTION': pred_train.ravel()}),
                        TARGET_COL, rmsle, 'RMSLE',
                        f"error_diagonal_RF_optuna.png", precision=5)

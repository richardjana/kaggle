import sys

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import mapk  # noqa
from kaggle_api_functions import submit_prediction
from competition_specifics import load_preprocess_data, TARGET_COL, COMPETITION_NAME


def xgb_feval_map3(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Custom evaluation function for XGBClassifier to compute MAP@3.
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Array of predictions.
    Returns:
        float: Metric value.
    """
    top_3 = np.argsort(-y_pred, axis=1)[:, :3]
    actual = [[int(label)] for label in y_true]

    return mapk(actual, top_3.tolist(), k=3)


def make_prediction(model: xgb.XGBClassifier, test_df: pd.DataFrame) -> None:
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
    submit_df.to_csv('predictions_XGB_optuna.csv',
                     columns=['id', TARGET_COL], index=False)


# Load dataset
dataframe, test, encoders = load_preprocess_data('XGB')

# Split into train/test sets
y = dataframe.pop(TARGET_COL).to_numpy()
X = dataframe.to_numpy()
categorical_features = [dataframe.columns.get_loc(c) for c in ['Soil Type', 'Crop Type']]

best_iterations = []

# Define objective function for Optuna
def objective(trial):
    best_iteration_folds = []

    param = {
        "objective": "multi:softprob",
        "eval_metric": xgb_feval_map3,
        "num_class": len(np.unique(y)),
        "tree_method": "hist",
        "verbosity": 0,
        "eta": trial.suggest_loguniform("learning_rate", 1e-3, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "n_estimators": 10000,
        "use_label_encoder": False,
        "early_stopping_rounds": 100
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mapa3_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**param)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        pred_val = model.predict_proba(X_val)
        top_3 = np.argsort(-pred_val, axis=1)[:, :3]
        actual = [[int(label)] for label in y_val]
        mapa3_scores.append(mapk(actual, top_3.tolist(), k=3))

        best_iteration_folds.append(model.get_booster().best_iteration)

    best_iterations.append(int(np.mean(best_iteration_folds)))

    return np.mean(mapa3_scores)

# Create and optimize Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000, timeout=60*60*11)

best_params = study.best_params
best_params.pop('n_estimators', best_iterations[study.best_trial.number])
best_params.pop('early_stopping_rounds', None)

# Print best trial
print('Best trial:')
print(f"  MAP@3: {study.best_value}")
print('  Best hyperparameters:')
for key, value in best_params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
model = xgb.XGBClassifier(**best_params, use_label_encoder=False)

eval_results = {}

model.fit(X, y)

joblib.dump(model, 'xgb_model.pkl')

# Evaluate
pred_train = np.asarray(model.predict_proba(X))
top_3 = np.argsort(-pred_train, axis=1)[:, :3]
actual = [[int(label)] for label in y]

map3_score = mapk(actual, top_3.tolist(), k=3)
print(f'MAP@3 score: {map3_score:.7f}')

# make prediction for the test data
make_prediction(model, test)

public_score = submit_prediction(COMPETITION_NAME, 'predictions_XGB_optuna.csv',
                                 f"XGB optuna ({study.best_value})")
print(f"Public score: {public_score:.7f}")

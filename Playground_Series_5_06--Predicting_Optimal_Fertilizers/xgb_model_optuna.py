import sys

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import mapk  # noqa
from kaggle_api_functions import submit_prediction
from competition_specifics import TARGET_COL, COMPETITION_NAME, load_preprocess_data

OPTUNA_FRAC = 0.25
N_AUGMENT = 4
try:
    SERIAL_NUMBER = sys.argv[1]
except IndexError:
    SERIAL_NUMBER = 0


def make_prediction(model: xgb.XGBClassifier, test_df: pd.DataFrame) -> None:
    """ Make a prediction for the test data, with a given model.
    Args:
        model (xgb.XGBClassifier): Model used for the prediction.
        test_df (pd.DataFrame): DataFrame with the test data, pre-processed.
    """
    submit_df = pd.read_csv('sample_submission.csv')
    pred_proba = np.asarray(model.predict_proba(test_df.to_numpy()))
    top_3_indices = np.argsort(-pred_proba, axis=1)[:, :3]

    top_3_labels = np.array([
        encoder.inverse_transform(sample_top3)
        for sample_top3 in top_3_indices
    ])

    joined_top3 = [' '.join(labels) for labels in top_3_labels]

    submit_df[TARGET_COL] = joined_top3
    submit_df.to_csv('predictions_XGB_optuna.csv',
                     columns=['id', TARGET_COL], index=False)


# Load dataset
train_full, test, X_original, encoder = load_preprocess_data()
for df in [train_full, test, X_original]:
    df = df.astype('category')

NUM_CLASSES = train_full[TARGET_COL].unique()

y_original = X_original.pop(TARGET_COL)

_, train = train_test_split(train_full, test_size=OPTUNA_FRAC, random_state=42)

best_iterations = []

# Define objective function for Optuna
def objective(trial):
    best_iteration_folds = []

    param = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': NUM_CLASSES,
        'tree_method': 'hist',
        'verbosity': 0,
        'eta': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'n_estimators': 10_000,
        "max_delta_step": trial.suggest_float('max_delta_step', 1e-3, 10, log=True),
        'gamma': trial.suggest_float('gamma', 1e-3, 10, log=True),
        'use_label_encoder': False,
        'early_stopping_rounds': 100,
        'enable_categorical': True
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mapa3_scores = []

    for train_idx, val_idx in skf.split(train, train[TARGET_COL]):
        X_train_fold, X_val_fold = train.iloc[train_idx].copy(), train.iloc[val_idx].copy()

        y_train_fold = X_train_fold.pop(TARGET_COL)
        y_val_fold = X_val_fold.pop(TARGET_COL)

        for k in range(int(N_AUGMENT * OPTUNA_FRAC)):
            X_orig_frac, _, y_orig_frac, _ = train_test_split(X_original, y_original,
                                                              test_size=1/5, random_state=k,
                                                              stratify=y_original)
            X_train_fold = pd.concat([X_train_fold, X_orig_frac], ignore_index=True)
            y_train_fold = pd.concat([y_train_fold, y_orig_frac], ignore_index=True)

        model = xgb.XGBClassifier(**param)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  verbose=False
                  )

        pred_val = model.predict_proba(X_val_fold)
        top_3 = np.argsort(-pred_val, axis=1)[:, :3]
        actual = [[int(label)] for label in y_val_fold]
        mapa3_scores.append(mapk(actual, top_3.tolist(), k=3))

        best_iteration_folds.append(model.get_booster().best_iteration)

    best_iterations.append(int(np.mean(best_iteration_folds)))

    return np.mean(mapa3_scores)

# Create and optimize Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10_000, timeout=60*60*5)

best_params = study.best_params
best_params['n_estimators'] = int(
    best_iterations[study.best_trial.number] * np.sqrt(1/OPTUNA_FRAC)
    )  # just a guess
best_params.pop('early_stopping_rounds', None)

# Print best trial
print(f"Best trial: {study.best_trial.number}/{len(study.trials)}")
print(f"  MAP@3: {study.best_value}")
print(f"  worst MAP@3: {min(trial.value for trial in study.trials if trial.value is not None)}")
print("  Best hyperparameters:")
for key, value in best_params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
y_full = train_full.pop(TARGET_COL)
for k in range(N_AUGMENT):
    X_orig_frac, _, y_orig_frac, _ = train_test_split(X_original, y_original, test_size=1/5,
                                                      random_state=k, stratify=y_original)
    train_full = pd.concat([train_full, X_orig_frac], ignore_index=True)
    y_full = pd.concat([y_full, y_orig_frac], ignore_index=True)

model = xgb.XGBClassifier(**best_params)
model.fit(train_full, y_full)
joblib.dump(model, 'xgb_model.pkl')

# make prediction for the test data
make_prediction(model, test)

public_score = submit_prediction(COMPETITION_NAME, 'predictions_XGB_optuna.csv',
                                 f"XGB optuna {SERIAL_NUMBER} ({study.best_value})")
print(f"Public score: {public_score}")

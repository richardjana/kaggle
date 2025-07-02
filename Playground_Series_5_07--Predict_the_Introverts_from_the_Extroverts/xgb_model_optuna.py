import sys
from typing import List

import joblib
import numpy as np
from numpy.typing import NDArray
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

sys.path.append('/'.join(__file__.split('/')[:-2]))
from competition_specifics import TARGET_COL, COMPETITION_NAME, load_preprocess_data
from kaggle_api_functions import submit_prediction
from kaggle_utilities import plot_confusion_matrix

try:
    SERIAL_NUMBER = sys.argv[1]
except IndexError:
    SERIAL_NUMBER = 0


def make_prediction(test_fold_preds: List[NDArray], encoder: LabelEncoder) -> None:
    """ Make a prediction for the test data, with a given model and ecoder for the target.
    Args:
        test_fold_preds (List[NDArray]): Predictions for each fold.
        encoder (LabelEncoder): Label encoder to recover the correct labels.
    """
    submit_df = pd.read_csv('sample_submission.csv')
    submit_df[TARGET_COL] = np.median(np.array(test_fold_preds), axis=0).astype(int)
    submit_df[TARGET_COL] = encoder.inverse_transform(submit_df[TARGET_COL])
    submit_df.to_csv('predictions_XGB_optuna.csv',
                     columns=['id', TARGET_COL], index=False)


# Load dataset
train, test, encoder = load_preprocess_data()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ADDITIONAL_PARAMS = {'early_stopping_rounds': 100,
                     'enable_categorical': True,
                     'eval_metric': 'logloss',
                     'n_estimators': 10_000,
                     'objective': 'binary:logistic',
                     'random_state': 77,
                     'tree_method': 'hist',
                     'use_label_encoder': False,
                     'verbosity': 0,
                     'scale_pos_weight': 2.8391709844559587
                     }


def objective(trial):
    """ Objective function for Optuna. """
    params = {'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
              'gamma': trial.suggest_float('gamma', 1e-3, 10, log=True),
              'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
              'max_delta_step': trial.suggest_float('max_delta_step', 1e-3, 10, log=True),
              'max_depth': trial.suggest_int('max_depth', 3, 20),
              'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
              'subsample': trial.suggest_float('subsample', 0.5, 1.0),
              'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
              'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
              }
    params.update(ADDITIONAL_PARAMS)

    oof_preds = np.zeros(len(train))
    oof_probas = np.zeros((len(train), 2))

    for train_idx, val_idx in skf.split(train, train[TARGET_COL]):
        X_train_fold, X_val_fold = train.iloc[train_idx].copy(), train.iloc[val_idx].copy()

        #X_train_fold = pd.concat([X_train_fold] * 2, ignore_index=True)

        y_train_fold = X_train_fold.pop(TARGET_COL)
        y_val_fold = X_val_fold.pop(TARGET_COL)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  verbose=False
                  )

        oof_preds[val_idx] = model.predict(X_val_fold)
        oof_probas[val_idx, :] = model.predict_proba(X_val_fold)

    #return accuracy_score(oof_preds, train[TARGET_COL])
    return roc_auc_score(train[TARGET_COL], oof_probas[:, 1])

# Create and optimize Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10, timeout=60*60*5)

best_params = study.best_params
best_params.update(ADDITIONAL_PARAMS)

# Print best trial
print(f"Best trial: {study.best_trial.number}/{len(study.trials)}")
print(f"  AUC: {study.best_value}")
print(f"  worst AUC: {min(trial.value for trial in study.trials if trial.value is not None)}")
print('  Best hyperparameters:')
for key, value in best_params.items():
    print(f"    {key}: {value}")


# Train final model with best parameters
oof_preds = np.zeros(len(train))
oof_probas = np.zeros((len(train), 2))
test_fold_preds = []

for train_idx, val_idx in skf.split(train, train[TARGET_COL]):
    X_train_fold, X_val_fold = train.iloc[train_idx].copy(), train.iloc[val_idx].copy()

    #X_train_fold = pd.concat([X_train_fold] * 2, ignore_index=True)

    y_train_fold = X_train_fold.pop(TARGET_COL)
    y_val_fold = X_val_fold.pop(TARGET_COL)

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              verbose=False
              )

    oof_preds[val_idx] = model.predict(X_val_fold)
    oof_probas[val_idx, :] = model.predict_proba(X_val_fold)

    test_fold_preds.append(model.predict(test))

    #importances = model.feature_importances_
    #feature_names = X_train_fold.columns
    #sorted_features = sorted(zip(importances, feature_names), reverse=True)


# save predictions for ensembling
joblib.dump({'oof_preds': oof_preds,
             'test_preds': np.mean(test_fold_preds, axis=0),
             'y_train': train[TARGET_COL].values
             },
             'stacking_data.pkl')

# make prediction for the test data
make_prediction(test_fold_preds, encoder)

final_accuracy = accuracy_score(train[TARGET_COL], oof_preds)
final_auc = roc_auc_score(train[TARGET_COL], oof_probas[:, 1])
print(final_accuracy, final_auc)

plot_confusion_matrix(train[TARGET_COL].to_numpy(), oof_preds.astype(int),
                      'confusion_matrix.png', encoder.classes_)

exit()
public_score = submit_prediction(COMPETITION_NAME, 'predictions_XGB_optuna.csv',
                                 f"XGB optuna {SERIAL_NUMBER} ({final_accuracy})")
print(f"Public score: {public_score}")

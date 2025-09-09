import sys

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_api_functions import submit_prediction

TARGET_COL = 'BeatsPerMinute'
COMPETITION_NAME = 'playground-series-s5e9'
N_FOLDS = 5

def make_prediction(model: XGBRegressor, test_df: pd.DataFrame) -> None:
    """ Make a prediction for the test data, with a given model.
    Args:
        modle (XGBRegressor): Model used for the prediction.
        test_df (pd.DataFrame): DataFrame with the test data.
    """
    submit_df = pd.read_csv('sample_submission.csv')

    submit_df[TARGET_COL] = model.predict(test_df)
    submit_df.to_csv('predictions_XGB_optuna.csv',
                     columns=['id', TARGET_COL], index=False)


def load_and_prepare(file_name: str, sep: str =',') -> pd.DataFrame:
    """ Read data from csv file and do some basic preprocessing.
    Args:
        file_name (str): Name of the csv file.
        sep (str): Separator for read_csv.
    Returns:
        pd.DataFrame: The created DataFrame.
    """
    df = pd.read_csv(file_name, sep=sep)
    try:  # train and test
        df.drop(columns='id', inplace=True)
    except KeyError:  # original data
        pass

    # FE ideas adopted from Chris Deotte
    for col in ['Energy', 'MoodScore', 'AcousticQuality']:
        for digit in range(1,10):
            df[f"{col}_d{digit}"] = ((df[col] * 10**digit) % 10).fillna(-1).astype('int8')

    for col in ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
                'TrackDurationMs', 'Energy']:
        for decimals in [8, 9]:
            df[f"{col}_r{decimals}"] = df[col].round(decimals)

    return df


def target_encode_with_original_data(df: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
    """ Use the original data for a type of target encoding.
    Args:
        df (pd.DataFrame): DataFrame to add the TE to.
        orig (pd.DataFrame): The original data.
    Returns:
        pd.DataFrame: DataFrame with added columns.
    """
    for col in [c for c in df.columns if c != TARGET_COL]:
        te_col = f"TEO_{col}"

        tmp_df = orig.groupby(col)[TARGET_COL].mean()
        tmp_df.name = te_col
        df = df.merge(tmp_df, on=col, how='left')

        df[te_col] = df[te_col].fillna(orig[TARGET_COL].mean())

    return df


# Load dataset
X_train = load_and_prepare('train.csv')
X_test = load_and_prepare('test.csv')
orig = load_and_prepare('original.csv', sep=';')

X_train = target_encode_with_original_data(X_train, orig)
X_test = target_encode_with_original_data(X_test, orig)

y_train = X_train.pop(TARGET_COL)


ADDITIONAL_PARAMS = {'objective': 'reg:squarederror',
                     'eval_metric': 'rmse',
                     'n_jobs': -1,
                     'random_state': 77,
                     'n_estimators': 10_000,
                     'early_stopping_rounds': 100,
                     'enable_categorical': True
                     }


kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
# Define objective function for Optuna
def objective(trial):
    params = {'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
              'max_depth': trial.suggest_int('max_depth', 3, 20),
              'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
              'subsample': trial.suggest_float('subsample', 0.6, 1.0),
              'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
              'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
              'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
              'max_delta_step': trial.suggest_float('max_delta_step', 0, 1.0),
              'gamma': trial.suggest_float('gamma', 0, 0.2)
              }
    params.update(ADDITIONAL_PARAMS)

    # Cross-validation
    rmses = []
    best_iteration_folds = []
    for train_idx, valid_idx in kf.split(X_train, y_train):
        X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # Train model
        model = XGBRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_valid_fold, y_valid_fold)],
                  verbose=False
                  )

        # Predict and evaluate
        pred = model.predict(X_valid_fold)
        rmses.append(root_mean_squared_error(y_valid_fold, pred))
        best_iteration_folds.append(model.best_iteration)

    trial.set_user_attr('n_estimators', int(np.mean(best_iteration_folds)
                                            *np.sqrt(N_FOLDS/(N_FOLDS-1))))

    return np.mean(rmses)


# Create and optimize Optuna study
study = optuna.create_study(direction='minimize',
                            study_name='bearsbeatbeets',
                            storage='sqlite:///optuna_study_xgb.db')
study.optimize(objective, n_trials=10_000, timeout=60*60*6)


# Train final model with best parameters
best_params = study.best_params
best_params.update(ADDITIONAL_PARAMS)
best_params['n_estimators'] = study.best_trial.user_attrs.get('n_estimators')
del best_params['early_stopping_rounds']


model = XGBRegressor(**best_params)
model.fit(X_train, y_train, verbose=False)

# make prediction for the test data
make_prediction(model, X_test)

public_score = submit_prediction(COMPETITION_NAME, 'predictions_XGB_optuna.csv',
                                 f"XGB optuna ({study.best_value})")
print(f'Public score: {public_score}')

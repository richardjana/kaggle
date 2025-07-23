import sys
from typing import List

import joblib
import numpy as np
from numpy.typing import NDArray
import optuna
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb

from collections.abc import Callable
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import RMSE

# strategy: predict the house price as a single value here, then use a model to translate this
# into an interval

# transform target with np.log1p ? -> np.expm1(preds)

# optional validation strategy: exclude the newest part of the data from training and use it only
# for extra validation --- this is meant to model that the test data contains rows that are newer
# than all the training data

# for 1-step prediction:
# LGBM: objective='quantile', alpha=0.95
# XGB: custom loss function

COMPETITION_NAME = 'prediction-interval-competition-ii-house-price'
TARGET_COL = 'sale_price'
NUM_COLS = ['Time_spent_Alone',  'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']
CAT_COLS = ['submarket', 'city', 'zoning', 'subdivision',
            'join_status', 'sale_warning', 'present_use']

OPTUNA_FRAC = 0.10  # run optuna on 10% of the training data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Apply cleaning operations to pandas DataFrame.
    Args:
        pd_df (pd.DataFrame): The DataFrame to clean.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df.drop('id', axis=1, inplace=True)
    except KeyError:
        pass

    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['year'] = df['sale_date'].dt.year
    df['month'] = df['sale_date'].dt.month
    df['day'] = df['sale_date'].dt.day

    latest_sale_date = df['sale_date'].max()
    time_differnces = latest_sale_date - df['sale_date']
    df['age_of_sale_days'] = time_differnces.dt.days
    df['age_of_sale_years'] = (time_differnces / np.timedelta64(1, 'D')) / 365.25

    df.drop('sale_date', axis=1, inplace=True)

    df['land_val'] = np.log1p(df['land_val'])
    df['imp_val'] = np.log1p(df['imp_val'])

    #df['val_ratio'] = df['land_val']/df['imp_val']
    df['pct_A'] = np.where((df['land_val'] + df['imp_val']) != 0, df['land_val'] / (df['land_val'] + df['imp_val']), 0.5)
    df['total_baths'] = df['bath_full'] + 0.75*df['bath_3qtr'] + 0.5*df['bath_half']
    #df['bath_to_beds'] = df['total_baths'] / df['beds']  # both can be 0

    # compare year renovated to year of sale or something?
    # total number of views

    df.drop('sale_nbr', axis=1, inplace=True)

    for col in ['submarket', 'subdivision']:  # unclear if this does anything
        df[col] = df[col].fillna('unknown').astype('category')

    try:
        df['sale_price'] = np.log1p(df['sale_price'])
    except KeyError:
        pass

    return df


def make_diagonal_plot(train: pd.DataFrame,
                       target_col: str,
                       metric: Callable[[List[float], List[float]], float],
                       metric_name: str,
                       fname: str,
                       precision: int = 2) -> None:
    """ Make a diagonal error plot, showing both training and validation data points.
    Args:
        train (pd.DataFrame): Training data.
        val (pd.DataFrame): Validation data.
        target_col (str): Name of the target column.
        metric (Callable[[List[float], List[float]], float]): Function to calculate the evaluation
            metric.
        metric_name (str): Name of the evaluation metric.
        fname (str): File name for the plot image.
        precision (int): Number of decimals to print for the metric. Defaults to 2.
    """
    chart = sns.scatterplot(data=train, x=target_col,
                            y='PREDICTION', alpha=0.25)

    min_val = min(chart.get_xlim()[0], chart.get_ylim()[0])
    max_val = max(chart.get_xlim()[1], chart.get_ylim()[1])
    chart.set_xlim([min_val, max_val])
    chart.set_ylim([min_val, max_val])
    chart.plot([min_val, max_val], [min_val, max_val], linewidth=1, color='k')

    chart.set_aspect('equal')
    chart.set_xlabel(target_col)
    chart.set_ylabel(f"Predicted {target_col}")

    metric_value = metric(train[target_col], train['PREDICTION'])
    labels = [f"training ({metric_value:.{precision}f})"]
    plt.legend(labels=labels, title=f"dataset ({metric_name}):", loc='best')

    plt.savefig(fname, bbox_inches='tight')
    plt.close()


# Load dataset
train_full = clean_data(pd.read_csv('dataset.csv'))
test = clean_data(pd.read_csv('test.csv'))

for df in [train_full, test]:
    for col in CAT_COLS:
        df[col] = df[col].astype('category')

_, train = train_test_split(train_full, test_size=OPTUNA_FRAC, random_state=42, shuffle=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)


ADDITIONAL_PARAMS = {'early_stopping_rounds': 100,
                     'enable_categorical': True,
                     'eval_metric': 'rmse',
                     'n_estimators': 10_000,
                     'objective': 'reg:squarederror',
                     'random_state': 77,
                     'tree_method': 'hist',
                     'use_label_encoder': False,
                     'verbosity': 0,
                     }


def objective(trial):
    """ Objective function for Optuna. """
    params = {'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
              'gamma': trial.suggest_float('gamma', 1e-3, 0.05, log=True),
              'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.03, log=True),
              'max_delta_step': trial.suggest_float('max_delta_step', 1e-3, 10, log=True),
              'max_depth': trial.suggest_int('max_depth', 3, 10),
              'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
              'subsample': trial.suggest_float('subsample', 0.4, 1.0),
              'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 3.0, log=True),
              'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True)
              }
    params.update(ADDITIONAL_PARAMS)

    oof_preds = np.zeros(len(train))

    for train_idx, val_idx in kf.split(train, train[TARGET_COL]):
        X_train_fold, X_val_fold = train.iloc[train_idx].copy(), train.iloc[val_idx].copy()

        y_train_fold = X_train_fold.pop(TARGET_COL)
        y_val_fold = X_val_fold.pop(TARGET_COL)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False)

        oof_preds[val_idx] = model.predict(X_val_fold)
    
    return RMSE(train[TARGET_COL], oof_preds, 6)


# Create and optimize Optuna study
study = optuna.create_study(study_name='house_price_mean',
                            direction='minimize',
                            storage='sqlite:///optuna_study_mean.db')
study.optimize(objective, n_trials=10_000, timeout=60*60*5)

best_params = study.best_params
best_params.update(ADDITIONAL_PARAMS)

# Print best trial
print(f"Best trial: {study.best_trial.number}/{len(study.trials)}")
print(f"  RMSE: {study.best_value}")
print(f"  worst RMSE: {max(trial.value for trial in study.trials if trial.value is not None)}")
print('  Best hyperparameters:')
for key, value in best_params.items():
    print(f"    {key}: {value}")


# Train final model with best parameters
oof_preds = np.zeros(len(train_full))
test_fold_preds = []

for train_idx, val_idx in kf.split(train_full, train_full[TARGET_COL]):
    X_train_fold, X_val_fold = train_full.iloc[train_idx].copy(), train_full.iloc[val_idx].copy()

    y_train_fold = X_train_fold.pop(TARGET_COL)
    y_val_fold = X_val_fold.pop(TARGET_COL)

    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              verbose=False)

    oof_preds[val_idx] = model.predict(X_val_fold)
    test_fold_preds.append(model.predict(test))


# save predictions for ensembling
joblib.dump({'oof_preds': oof_preds,
             'test_preds': np.mean(test_fold_preds, axis=0),
             'y_train': train_full[TARGET_COL].values
             },
             'stacking_data_mean.pkl')


train_full['PREDICTION'] = oof_preds
make_diagonal_plot(train_full, target_col=TARGET_COL, metric=RMSE,
                   metric_name='RMSE', fname='diagonal_plot_mean.png')

import sys
from typing import List

import joblib
import numpy as np
from numpy.typing import NDArray
import optuna
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb

from collections.abc import Callable
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import RMSE


COMPETITION_NAME = 'prediction-interval-competition-ii-house-price'
TARGET_COL = 'sale_price'
NEW_TARGET_COL = 'residual'
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

    df['pct_A'] = np.where((df['land_val'] + df['imp_val']) != 0,
                           df['land_val'] / (df['land_val'] + df['imp_val']), 0.5)
    df['total_baths'] = df['bath_full'] + 0.75*df['bath_3qtr'] + 0.5*df['bath_half']

    df.drop('sale_nbr', axis=1, inplace=True)

    for col in ['submarket', 'subdivision']:
        df[col] = df[col].fillna('unknown').astype('category')

    return df


def make_prediction(test_means: NDArray, test_fold_preds: List[NDArray], gamma: float) -> None:
    """ Make a prediction for the test data, constructing the interval from the mean value and
        residuals.
    Args:
        test_means (NDArray): Means predicted by the first stage model.
        test_fold_preds (List[NDArray]): Residuals predicted by the second stage model.
        gamma (float): Optimized gamma parameter to construct the intervals.
    """
    submit_df = pd.read_csv('sample_submission.csv')
    submit_df['pi_lower'] = test_means - gamma * np.sqrt(np.mean(test_fold_preds, axis=0))
    submit_df['pi_upper'] = test_means + gamma * np.sqrt(np.mean(test_fold_preds, axis=0))
    submit_df.to_csv('predictions_XGB_optuna--interval.csv',
                     columns=['id', 'pi_lower', 'pi_upper'], index=False)


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


def winkler_score(y_true, lower, upper, alpha=0.1, return_coverage=False):
    """Compute the Winkler Interval Score for prediction intervals.
    Args:
        y_true (array-like): True observed values.
        lower (array-like): Lower bounds of prediction intervals.
        upper (array-like): Upper bounds of prediction intervals.
        alpha (float): Significance level (e.g., 0.1 for 90% intervals).
        return_coverage (bool): If True, also return empirical coverage.
    Returns:
        score (float): Mean Winkler Score.
        coverage (float, optional): Proportion of true values within intervals.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    score += np.where(y_true < lower, penalty_lower, 0)
    score += np.where(y_true > upper, penalty_upper, 0)

    if return_coverage:
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(inside)
        return np.mean(score), coverage

    return np.mean(score)


def winkler_for_gamma(gamma: float, y_true: NDArray, pred: NDArray,
                      pred_err: NDArray, alpha: float = 0.1) -> float:
    """ Construct the interval and calculate the Winkler score from the given parameters.
    Args:
        gamma (float): Interval width parameter.
        y_true (NDArray): Actual target values.
        pred (NDArray): Predicted target values.
        pred_err (NDArray): Predicted target residuals.
        alpha (float, optional): Significance level. Defaults to 0.1.
    Returns:
        float: Winkler score calculated from the inputs.
    """
    lower = pred - gamma * np.sqrt(pred_err)
    upper = pred + gamma * np.sqrt(pred_err)
    return winkler_score(y_true, lower, upper, alpha=alpha)


# load data from first stage
data = joblib.load('stacking_data_mean.pkl')
test_mean = data['test_preds']

# Load dataset, replace original TAREGT_COL with residual
train_full = clean_data(pd.read_csv('dataset.csv'))
train_full[NEW_TARGET_COL] = np.log1p((train_full[TARGET_COL] - data['oof_preds']) ** 2)
train_full['predicted_mean'] = data['oof_preds']
test = clean_data(pd.read_csv('test.csv'))

for df in [train_full, test]:
    for col in CAT_COLS:
        df[col] = df[col].astype('category')

_, train = train_test_split(train_full, test_size=OPTUNA_FRAC, random_state=42, shuffle=True)
train_mean = train.pop(TARGET_COL)
train_mean_predicted = train.pop('predicted_mean')

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
    params = {'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.6),
              'gamma': trial.suggest_float('gamma', 1e-3, 1, log=True),
              'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.0075, log=True),
              'max_delta_step': trial.suggest_float('max_delta_step', 0.3, 1, log=True),
              'max_depth': trial.suggest_int('max_depth', 6, 20),
              'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),
              'subsample': trial.suggest_float('subsample', 0.65, 0.9),
              'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 2.0, log=True),
              'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True)
              }
    params.update(ADDITIONAL_PARAMS)

    oof_preds = np.zeros(len(train))

    for train_idx, val_idx in kf.split(train, train[NEW_TARGET_COL]):
        X_train_fold, X_val_fold = train.iloc[train_idx].copy(), train.iloc[val_idx].copy()

        y_train_fold = X_train_fold.pop(NEW_TARGET_COL)
        y_val_fold = X_val_fold.pop(NEW_TARGET_COL)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False)

        oof_preds[val_idx] = np.expm1(model.predict(X_val_fold))

    oof_preds = np.clip(oof_preds, 1e-6, None)

    res = minimize_scalar(winkler_for_gamma, bounds=(0.1, 10.0), method='bounded',
                           args=(train_mean, train_mean_predicted, oof_preds))

    return res.fun


# Create and optimize Optuna study
study = optuna.create_study(study_name='house_price_residual',
                            direction='minimize',
                            storage='sqlite:///optuna_study_residual.db')
study.optimize(objective, n_trials=10_000, timeout=60*60*5)

best_params = study.best_params
best_params.update(ADDITIONAL_PARAMS)

# Print best trial
print(f"Best trial: {study.best_trial.number}/{len(study.trials)}")
print(f"  Winkler score: {study.best_value}")
print('  Best hyperparameters:')
for key, value in best_params.items():
    print(f"    {key}: {value}")


# Train final model with best parameters
oof_preds = np.zeros(len(train_full))
test_fold_preds = []
train_full_mean = train_full.pop(TARGET_COL)
train_full_mean_predicted = train_full.pop('predicted_mean')

for train_idx, val_idx in kf.split(train_full, train_full[NEW_TARGET_COL]):
    X_train_fold, X_val_fold = train_full.iloc[train_idx].copy(), train_full.iloc[val_idx].copy()

    y_train_fold = X_train_fold.pop(NEW_TARGET_COL)
    y_val_fold = X_val_fold.pop(NEW_TARGET_COL)

    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              verbose=False)

    oof_preds[val_idx] = np.expm1(model.predict(X_val_fold))
    test_fold_preds.append(np.expm1(model.predict(test)))

oof_preds = np.clip(oof_preds, 1e-6, None)
test_fold_preds = np.clip(test_fold_preds, 1e-6, None)

res = minimize_scalar(winkler_for_gamma, bounds=(0.1, 10.0), method='bounded',
                      args=(train_full_mean, train_full_mean_predicted, oof_preds))
print(f"Final Winkler score {res.fun:.5f}, with gamma {res.x:.5f}")

# save predictions for ensembling
joblib.dump({'oof_preds': oof_preds,
             'test_preds': np.mean(test_fold_preds, axis=0),
             'y_train': train_full[NEW_TARGET_COL].values
             },
             'stacking_data_residual.pkl')

# make prediction for the test data
make_prediction(test_mean, test_fold_preds, res.x)

train_full['PREDICTION'] = oof_preds
train_full[NEW_TARGET_COL] = np.expm1(train_full[NEW_TARGET_COL])
make_diagonal_plot(train_full, target_col=NEW_TARGET_COL, metric=RMSE,
                   metric_name='RMSE', fname='diagonal_plot_residual.png')

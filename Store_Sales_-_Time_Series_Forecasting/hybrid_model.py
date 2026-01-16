import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import pacf
from xgboost import XGBRegressor

# design choice: a single (hybrid) model for all data, or separate models for each product
# category, or separate models for each category/store combination

# TODO: create statistical features on product families and store locations

'''
XGB_PARAMS = {'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_jobs': -1,
            'random_state': 77,
            'n_estimators': 10_000,
            'early_stopping_rounds': 100,
            'enable_categorical': True,
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
              'max_depth': trial.suggest_int('max_depth', 3, 20),
              'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
              'subsample': trial.suggest_float('subsample', 0.6, 1.0),
              'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
              'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
              'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
              'max_delta_step': trial.suggest_float('max_delta_step', 0, 1.0),
              'gamma': trial.suggest_float('gamma', 0, 0.2)
              }
'''
XGB_PARAMS = {'enable_categorical': True}


#for linear model, select lags programatically, or all the same fixed_lags = [1, 2, 3, 7, 14, 30]
def select_lags(y, threshold=0.1, max_lag=40):
    pacf_vals = pacf(y, nlags=max_lag)
    return [i for i, val in enumerate(pacf_vals) if i != 0 and abs(val) > threshold]


class HybridTimeSeriesPipeline:
    def __init__(self, horizon=16):
        self.base_model = LinearRegression()
        self.booster_model = MultiOutputRegressor(XGBRegressor(**XGB_PARAMS))
        self.horizon = horizon
        self._is_fitted = False
        self.scaler = StandardScaler()

    def create_targets(self, y):
        return pd.DataFrame({
            f"y_t+{i+1}": y.shift(-i-1)
            for i in range(self.horizon)
        }).add_prefix('target_')

    def make_base_features(self, df, y, lags=[1, 2, 3, 7, 14], rolling_windows=[7, 14]):
        base = pd.DataFrame(index=df.index)

        # Lag features
        for lag in lags:
            base[f"lag_{lag}"] = y.shift(lag)

        # Rolling means
        for window in rolling_windows:
            base[f"roll_mean_{window}"] = y.shift(1).rolling(window).mean()
            # min
            # max
            # std_dev

        # Time features
        base['dayofweek'] = df.index.dayofweek
        base['month'] = df.index.month
        base['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0

        # flags weekend / holiday
        base['is_weekend'] = df.index.dayofweek >= 5  # Saturday=5, Sunday=6
        # holidays_events.csv

        # Fourier terms (seasonality)
        base['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        base['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        # 7
        # 30

        return base.add_prefix('base_')

    def make_boost_features(self, df, y, lags=range(1, 31), rolling_windows=[7, 14, 30]):
        boost = pd.DataFrame(index=df.index)

        # Extended lag features
        for lag in lags:
            boost[f"lag_{lag}"] = y.shift(lag)

        # Rolling stats
        for window in rolling_windows:
            boost[f"roll_std_{window}"] = y.shift(1).rolling(window).std()
            boost[f"roll_min_{window}"] = y.shift(1).rolling(window).min()
            boost[f"roll_max_{window}"] = y.shift(1).rolling(window).max()
            boost[f"roll_q25_{window}"] = y.shift(1).rolling(window).quantile(0.25)
            boost[f"roll_q75_{window}"] = y.shift(1).rolling(window).quantile(0.75)

        # Interaction terms
        boost['lag_1_x_dayofweek'] = y.shift(1) * df.index.dayofweek

        # External regressors
        for col in df.columns:
            boost[col] = df[col]

        return boost.add_prefix('boost_')

    def fit(self, df, y):
        Y = self.create_targets(y)
        X_base = self.make_base_features(df, y)
        X_boost = self.make_boost_features(df, y)

        # Align and clean
        full = pd.concat([X_base, X_boost, Y], axis=1).dropna()
        X_base_clean = full[X_base.columns]
        X_boost_clean = full[X_boost.columns]
        Y_clean = full[Y.columns]

        # Scale base features
        X_base_scaled = self.scaler.fit_transform(X_base_clean)

        # Fit base model
        self.base_model.fit(X_base_scaled, Y_clean)
        Y_base_pred = self.base_model.predict(X_base_scaled)

        # Fit booster on residuals
        Y_resid = Y_clean - Y_base_pred
        self.booster_model.fit(X_boost_clean, Y_resid)

        self._is_fitted = True

    def predict(self, df, y):
        if not self._is_fitted:
            raise ValueError('Model must be fitted first.')

        X_base = self.make_base_features(df, y)
        X_boost = self.make_boost_features(df, y)

        # Align and clean
        full = pd.concat([X_base, X_boost], axis=1).dropna()
        X_base_clean = full[X_base.columns]
        X_boost_clean = full[X_boost.columns]

        # Scale base features
        X_base_scaled = self.scaler.transform(X_base_clean)

        # Predict
        Y_base_pred = self.base_model.predict(X_base_scaled)
        Y_boost_pred = self.booster_model.predict(X_boost_clean)

        return Y_base_pred + Y_boost_pred


def spaced_multi_horizon_splits(n_samples, horizon=16, n_folds=5, min_train_size=200):
    """
    Returns a list of (train_idx, val_idx) where:
      - train is [0, train_end)
      - val   is [train_end, train_end + horizon)
    Train_end values are spaced between min_train_size and n_samples - horizon.
    """
    max_train_end = n_samples - horizon
    if max_train_end <= min_train_size:
        raise ValueError('Not enough data for given min_train_size and horizon.')

    # Choose n_folds train_end points, spaced across the usable range
    train_ends = np.linspace(min_train_size, max_train_end, n_folds, dtype=int)

    splits = []
    for train_end in train_ends:
        val_start = train_end
        val_end = train_end + horizon
        if val_end > n_samples:
            continue

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))

    return splits


TARGET_COL = 'sales'

X = pd.read_csv('train.csv', parse_dates=['date'])
X = X.set_index('date').sort_index()
X['family'] = X['family'].astype('category')
y = X.pop(TARGET_COL)


results = []
for fold, (train_idx, val_idx) in enumerate(spaced_multi_horizon_splits(len(X))):
    model = HybridTimeSeriesPipeline(horizon=16)
    model.fit(X.loc[X.index[train_idx]],
              y.loc[y.index[train_idx]])

    # predict on full history (possibly more data than necessary?)
    X_hist = X.loc[X.index[:val_idx[-1] + 1]]
    y_hist = y.loc[y.index[:val_idx[-1] + 1]]
    preds_full = model.predict(X_hist, y_hist)
    preds = preds_full[-len(val_idx):]

    # compute metric
    Y_true = model.create_targets(y_hist).iloc[val_idx]
    results.append(mean_absolute_error(preds, Y_true.values))
    print(f"Fold {fold+1}: MAE={results[-1]:.4f}")

''' change feature engineering like so:
def make_base_features(self, df, y, ...):
    out = []

    for (store, family), group in df.groupby(["store_nbr", "family"]):
        y_group = y.loc[group.index]

        base = pd.DataFrame(index=group.index)
        base["lag_1"] = y_group.shift(1)
        base["lag_2"] = y_group.shift(2)
        base["roll_mean_7"] = y_group.shift(1).rolling(7).mean()
        ...
        out.append(base)

    return pd.concat(out).sort_index()
'''

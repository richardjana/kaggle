import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import pacf
from xgboost import XGBRegressor


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
        out = []

        for (store, family), group in df.groupby(level=[0, 1], observed=True):
            y_group = y.loc[group.index]
            base = pd.DataFrame(index=group.index)

            # Trend
            base['t_group'] = np.arange(len(base), dtype=float)

            # Lag features
            for lag in lags:
                base[f"lag_{lag}"] = y_group.shift(lag)

            # Rolling means
            for window in rolling_windows:
                base[f"roll_mean_{window}"] = y_group.shift(1).rolling(window).mean()
                # min
                # max
                # std_dev

            # Time features
            dates = group.index.get_level_values('date')
            base['dayofweek'] = dates.dayofweek
            base['month'] = dates.month
            base['hour'] = dates.hour if hasattr(dates, 'hour') else 0

            # flags weekend / holiday
            base['is_weekend'] = (dates.dayofweek >= 5).astype(bool)  # Saturday=5, Sunday=6
            # holidays_events.csv

            # Fourier terms (seasonality)
            base['sin_day'] = np.sin(2 * np.pi * dates.dayofyear / 365)
            base['cos_day'] = np.cos(2 * np.pi * dates.dayofyear / 365)
            # 7
            # 30

            out.append(base)

        return pd.concat(out).sort_index().add_prefix('base_')

    def make_boost_features(self, df, y, lags=range(1, 31), rolling_windows=[7, 14, 30]):
        out = []

        for (store, family), group in df.groupby(level=[0, 1], observed=True):
            y_group = y.loc[group.index]
            boost = pd.DataFrame(index=group.index)

            # Extended lag features
            for lag in lags:
                boost[f"lag_{lag}"] = y_group.shift(lag)

            # Rolling stats
            for window in rolling_windows:
                boost[f"roll_std_{window}"] = y_group.shift(1).rolling(window).std()
                boost[f"roll_min_{window}"] = y_group.shift(1).rolling(window).min()
                boost[f"roll_max_{window}"] = y_group.shift(1).rolling(window).max()
                boost[f"roll_q25_{window}"] = y_group.shift(1).rolling(window).quantile(0.25)
                boost[f"roll_q75_{window}"] = y_group.shift(1).rolling(window).quantile(0.75)

            # Interaction terms
            boost['lag_1_x_dayofweek'] = (y_group.shift(1)
                                          * group.index.get_level_values('date').dayofweek)

            # join external regressors
            boost = boost.join(group, how='left')

            out.append(boost)

        boost = pd.concat(out).sort_index()

        return boost.add_prefix('boost_')

    def fit(self, df, y):
        Y = self.create_targets(y)
        X_base = self.make_base_features(df, y)
        X_boost = self.make_boost_features(df, y)

        # Align and clean (Do I really need this? If no, move the next few sections into the
        # make_features functions.)
        full = pd.concat([X_base, X_boost, Y], axis=1).dropna()
        X_base_clean = full[X_base.columns]
        X_boost_clean = full[X_boost.columns]
        Y_clean = full[Y.columns]

        # Flatten index for model input
        X_base_clean = X_base_clean.reset_index()
        X_boost_clean = X_boost_clean.reset_index()

        # store_nbr and family need encoding for base model!
        X_base_clean = X_base_clean.drop(columns=['date'])
        cat_cols = ['store_nbr', 'family']
        num_cols = X_base_clean.columns.difference(cat_cols)
        X_base_num = X_base_clean[num_cols]
        X_base_cat = pd.get_dummies(X_base_clean[cat_cols], drop_first=False)
        X_base_clean = pd.concat([X_base_num, X_base_cat], axis=1)

        X_boost_clean['family'] = X_boost_clean['family'].astype('category')
        X_boost_clean['store_nbr'] = X_boost_clean['store_nbr'].astype('category')
        X_boost_clean = X_boost_clean.drop(columns=['date'])

        # Scale base features
        X_base_scaled = X_base_clean.copy()
        scale_cols = (X_base_clean
                      .select_dtypes(include=['number'])
                      .columns
                      .difference(X_base_clean.select_dtypes(include=['bool']).columns))
        X_base_scaled[scale_cols] = self.scaler.fit_transform(X_base_clean[scale_cols])

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

        # Flatten index for model input
        X_base_clean = X_base_clean.reset_index()
        X_boost_clean = X_boost_clean.reset_index()

        X_base_clean = X_base_clean.drop(columns=['date'])
        cat_cols = ['store_nbr', 'family']
        num_cols = X_base_clean.columns.difference(cat_cols)
        X_base_num = X_base_clean[num_cols]
        X_base_cat = pd.get_dummies(X_base_clean[cat_cols], drop_first=False)
        X_base_clean = pd.concat([X_base_num, X_base_cat], axis=1)

        X_boost_clean['family'] = X_boost_clean['family'].astype('category')
        X_boost_clean['store_nbr'] = X_boost_clean['store_nbr'].astype('category')
        X_boost_clean = X_boost_clean.drop(columns=['date'])

        # Scale base features
        X_base_scaled = X_base_clean.copy()
        scale_cols = (X_base_clean
                      .select_dtypes(include=['number'])
                      .columns
                      .difference(X_base_clean.select_dtypes(include=['bool']).columns))
        X_base_scaled[scale_cols] = self.scaler.transform(X_base_clean[scale_cols])

        # Predict
        Y_base_pred = self.base_model.predict(X_base_scaled)
        Y_boost_pred = self.booster_model.predict(X_boost_clean)

        return pd.DataFrame(
            Y_base_pred + Y_boost_pred,
            index=full.index,
            columns=self.create_targets(pd.Series(index=full.index, dtype=float)).columns)

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

def panel_date_splits(df, horizon=16, n_folds=5, min_train_days=200, history_days=14):
    """
    Panel-aware date-based CV.

    Each split:
      - train: all rows with date <= cutoff
      - val: rows covering [cutoff - history_days + 1, cutoff + horizon]
             (history included for feature generation)

    Returns list of (train_idx, val_idx) using positional indices.
    """
    date_level='date'

    dates = df.index.get_level_values(date_level).unique().sort_values()

    if len(dates) < min_train_days + horizon:
        raise ValueError('Not enough dates for requested split.')

    cutoff_positions = np.linspace(min_train_days,
                                   len(dates) - horizon,
                                   n_folds,
                                   dtype=int)

    date_index = df.index.get_level_values(date_level)
    splits = []

    for pos in cutoff_positions:
        train_end = dates[pos]

        # Validation ranges
        hist_start = dates[max(0, pos - history_days + 1)]
        val_end    = dates[pos + horizon - 1]

        # Masks
        train_mask = date_index <= train_end

        val_mask = (date_index > train_end) & (date_index <= val_end)
        hist_mask = (date_index >= hist_start) & (date_index <= val_end)

        train_idx = np.flatnonzero(train_mask)
        val_idx   = np.flatnonzero(hist_mask)
        eval_idx  = np.flatnonzero(val_mask)

        splits.append((train_idx, val_idx, eval_idx))

    return splits


TARGET_COL = 'sales'
HORIZON = 16
HISTORY = 31  # max(max(lags), max(rolling_windows))

X = pd.read_csv('train.csv', parse_dates=['date'])
X['family'] = X['family'].astype('category')
X = X.set_index(['store_nbr', 'family', 'date']).sort_index()
y = X.pop(TARGET_COL)

results = []

splits = panel_date_splits(X,
                           horizon=HORIZON,
                           n_folds=5,
                           min_train_days=200,
                           history_days=HISTORY)

for fold, (train_idx, val_idx, eval_idx) in enumerate(splits):
    model = HybridTimeSeriesPipeline(horizon=HORIZON)

    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    preds = model.predict(X.iloc[val_idx], y.iloc[val_idx])

    Y_true = model.create_targets(y.iloc[eval_idx])
    Y_pred = preds.reindex(Y_true.index)

    # Keep only rows where all horizons exist
    valid_mask = ~Y_true.isna().any(axis=1)
    Y_true_valid = Y_true.loc[valid_mask]
    Y_pred_valid = Y_pred.loc[valid_mask]

    assert Y_pred_valid.shape == Y_true_valid.shape
    assert np.isfinite(Y_pred_valid.values).all()
    assert np.isfinite(Y_true_valid.values).all()

    mae = mean_absolute_error(Y_pred_valid.values, Y_true_valid.values)

    print(f"Fold {fold+1}: MAE={mae:.4f}")

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append('../')
from kaggle_utilities import make_category_error_plot, make_ROC_plot, min_max_scaler

target_col = 'rainfall'

def clean_data(pd_df): # clean dataset
    # day column behaves suspiciously
    pd_df['day'] = pd_df.apply(lambda row: (row.id + pd_df['day'][0]) % 365, axis=1)

    pd_df.drop('id', axis=1, inplace=True)

    # 1 missing value in 'winddirection'
    pd_df['winddirection'].fillna(pd_df['winddirection'].mean(), inplace=True)

    # replace day / winddirection with cyclic representation
    pd_df['day_sin'] = pd_df.apply(lambda row: np.sin(2*np.pi*row.day/365), axis=1)
    pd_df['day_cos'] = pd_df.apply(lambda row: np.cos(2*np.pi*row.day/365), axis=1)
    pd_df.drop('day', axis=1, inplace=True)
    pd_df['winddirection_sin'] = pd_df.apply(lambda row: np.sin(2*np.pi*row.winddirection/360), axis=1)
    pd_df['winddirection_cos'] = pd_df.apply(lambda row: np.cos(2*np.pi*row.winddirection/360), axis=1)
    pd_df.drop('winddirection', axis=1, inplace=True)

    ### feature engineering ###
    pd_df['cloud_sunshine_ratio'] = pd_df['cloud'] / pd_df['sunshine'].clip(lower=0.1)
    pd_df['cloud_sunshine_interaction'] = pd_df['cloud'] * pd_df['sunshine']
    pd_df['dewpoint_depression'] = pd_df['temperature'] - pd_df['dewpoint']

    '''# Compute temperature range
    pd_df['temp_range'] = pd_df['maxtemp'] - pd_df['mintemp']

    # Compute dew point depression
    pd_df['dewpoint_depression'] = pd_df['temperature'] - pd_df['dewpoint']

    # Compute pressure change from previous day
    pd_df['pressure_change'] = pd_df['pressure'].diff().fillna(0)

    # Compute humidity to dew point ratio
    pd_df['humidity_dewpoint_ratio'] = pd_df['humidity'] / pd_df['dewpoint'].clip(lower=0.1)

    # Compute cloud coverage to sunshine ratio
    pd_df['cloud_sunshine_ratio'] = pd_df['cloud'] / pd_df['sunshine'].clip(lower=0.1)

    # Compute wind intensity factor
    pd_df['wind_humidity_factor'] = pd_df['windspeed'] * (pd_df['humidity'] / 100)

    # Compute temperature-humidity index
    pd_df['temp_humidity_index'] = (
        0.8 * pd_df['temperature']
        + (pd_df['humidity'] / 100) * (pd_df['temperature'] - 14.3)
        + 46.4
    )

    # Compute pressure acceleration
    pd_df['pressure_acceleration'] = pd_df['pressure_change'].diff().fillna(0)

    # Derive month feature
    pd_df['month'] = ((pd_df['day'] - 1) // 30) + 1
    pd_df['month'] = pd_df['month'].clip(upper=12)

    # Convert day to season
    pd_df['season'] = ((pd_df['month'] - 1) // 3) + 1

    # Capture cyclical nature of days in a year
    pd_df['day_of_year_sin'] = np.sin(2 * np.pi * pd_df['day'] / 365)
    pd_df['day_of_year_cos'] = np.cos(2 * np.pi * pd_df['day'] / 365)

    # Compute rolling averages for selected windows
    for window in [3, 7, 14]:
        pd_df[f'temperature_rolling_{window}d'] = pd_df['temperature'].rolling(
            window=window, min_periods=1
        ).mean()
        pd_df[f'pressure_rolling_{window}d'] = pd_df['pressure'].rolling(
            window=window, min_periods=1
        ).mean()
        pd_df[f'humidity_rolling_{window}d'] = pd_df['humidity'].rolling(
            window=window, min_periods=1
        ).mean()
        pd_df[f'cloud_rolling_{window}d'] = pd_df['cloud'].rolling(
            window=window, min_periods=1
        ).mean()
        pd_df[f'windspeed_rolling_{window}d'] = pd_df['windspeed'].rolling(
            window=window, min_periods=1
        ).mean()

    # Compute short-term trends (3-day)
    pd_df['temp_trend_3d'] = pd_df['temperature'].diff(3).fillna(0)
    pd_df['pressure_trend_3d'] = pd_df['pressure'].diff(3).fillna(0)
    pd_df['humidity_trend_3d'] = pd_df['humidity'].diff(3).fillna(0)

    # Define extreme weather indicators
    pd_df['extreme_temp'] = (
        (pd_df['temperature'] > pd_df['temperature'].quantile(0.95))
        | (pd_df['temperature'] < pd_df['temperature'].quantile(0.05))
    ).astype(int)
    pd_df['extreme_humidity'] = (
        (pd_df['humidity'] > pd_df['humidity'].quantile(0.95))
        | (pd_df['humidity'] < pd_df['humidity'].quantile(0.05))
    ).astype(int)
    pd_df['extreme_pressure'] = (
        (pd_df['pressure'] > pd_df['pressure'].quantile(0.95))
        | (pd_df['pressure'] < pd_df['pressure'].quantile(0.05))
    ).astype(int)

    # Create interaction terms
    pd_df['temp_humidity_interaction'] = pd_df['temperature'] * pd_df['humidity']
    pd_df['pressure_wind_interaction'] = pd_df['pressure'] * pd_df['windspeed']
    pd_df['cloud_sunshine_interaction'] = pd_df['cloud'] * pd_df['sunshine']
    pd_df['dewpoint_humidity_interaction'] = pd_df['dewpoint'] * pd_df['humidity']

    # Compute moving standard deviations for variability analysis
    for window in [7, 14]:
        pd_df[f'temp_std_{window}d'] = pd_df['temperature'].rolling(
            window=window, min_periods=4
        ).std().fillna(0)
        pd_df[f'pressure_std_{window}d'] = pd_df['pressure'].rolling(
            window=window, min_periods=4
        ).std().fillna(0)
        pd_df[f'humidity_std_{window}d'] = pd_df['humidity'].rolling(
            window=window, min_periods=4
        ).std().fillna(0)'''

    return pd_df

##### load data,  split into train / validation / test #####
dataframe = clean_data(pd.read_csv('train.csv').rename(columns={'temparature': 'temperature'}))

# original dataset #
orig = pd.read_csv('Rainfall.csv').rename(columns={'temparature': 'temperature'})
orig['rainfall'] = orig['rainfall'].map({'yes': 1, 'no': 0})
orig['id'] = orig.index
orig.columns = [column.strip() for column in orig.columns]
orig.dropna(axis=0, how='any', inplace=True)
orig = clean_data(orig)
dataframe = pd.concat([dataframe, orig], axis=0)
dataframe.reset_index(inplace=True, drop=True)

#dataframe, rest = train_test_split(dataframe, test_size=0.80) # reduce dataset size for testing
train, val = train_test_split(dataframe, test_size=0.2)
test = clean_data(pd.read_csv('test.csv').rename(columns={'temparature': 'temperature'}))

### scale columns (not cyclical representations, not target column) ###
scale_columns = [col for col in train.keys() if (col[-4:] not in ['_sin', '_cos']) and (col != target_col)]
train, val, test = min_max_scaler([train, val, test], scale_columns)

def fit_linear_model(X, y):
    model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1, tol=1e-5, max_iter=100_000)
    model.fit(X, y)

    return model

def x_y_pd_dataframe(pd_df):
    y = pd_df[target_col].to_numpy()
    x_df = pd_df.drop(target_col, axis=1)
    X = x_df.to_numpy()

    return X, y

# train linear model
X, y = x_y_pd_dataframe(train)
model = fit_linear_model(X, y)

# make predictions on train / test set
train['PREDICTION'] = model.predict(X)
train['PREDICTION_PROBABILITY'] = model.predict_proba(X)[::,1]
print("Training accuracy:", accuracy_score(train[target_col], train['PREDICTION']))
print('Training F1:', f1_score(train[target_col], train['PREDICTION']))
print('Training ROC AUC:', sklearn.metrics.roc_auc_score(train[target_col], train['PREDICTION_PROBABILITY']))
X, y = x_y_pd_dataframe(val)
val['PREDICTION'] = model.predict(X)
val['PREDICTION_PROBABILITY'] = model.predict_proba(X)[::,1]
print("Validation accuracy:", accuracy_score(val[target_col], val['PREDICTION']))
print('Validation F1:', f1_score(val[target_col], val['PREDICTION']))
print('Validation ROC AUC:', sklearn.metrics.roc_auc_score(val[target_col], val['PREDICTION_PROBABILITY']))

train.rename(columns={'pressure': 'id'}, inplace=True)
val.rename(columns={'pressure': 'id'}, inplace=True)
make_category_error_plot(train, target_col, 'category_error_training.png', 2)
make_category_error_plot(val, target_col, 'category_error_validation.png', 2)

make_ROC_plot(train, target_col, 'ROC_training.png')
make_ROC_plot(val, target_col, 'ROC_validation.png')

X_test = test.to_numpy()
prediction = model.predict(X_test)
test = pd.read_csv('test.csv')
test[target_col] = prediction
test.to_csv(f"predictions_regression.csv", columns=['id', target_col], index=False)

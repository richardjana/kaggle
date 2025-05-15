import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import early_stopping, log_evaluation

import sys
import keras
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

sys.path.append('../')
from kaggle_utilities import min_max_scaler, make_training_plot, make_diagonal_plot, rmsle, rmsle_metric  # noqa

TARGET_COL = 'Calories'

def lgb_rmsle(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return 'rmsle', np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2)), False


def clean_data(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ Apply cleaning operations to pandas DataFrame.
    Args:
        pd_df (pd.DataFrame): The DataFrame to clean.
        drop (bool, optional): If rows should be dropped (option for training DataFrame) or not
            (test DataFrame). Defaults to True.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    pd_df.drop('id', axis=1, inplace=True)

    pd_df['Sex'] = pd_df['Sex'].map({'male': 0, 'female': 1})

    return pd_df

# Load dataset
#data = fetch_california_housing()
#X, y = data.data, data.target
dataframe = clean_data(pd.read_csv('train.csv'))
#dataframe, rest = train_test_split(dataframe, test_size=0.95)  # reduce dataset size for testing
test = clean_data(pd.read_csv('test.csv'))

# Split into train/test sets
y = dataframe.pop(TARGET_COL).to_numpy()
X = dataframe.to_numpy()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=100,
    learning_rate=0.1
)

# Train with proper early stopping and logging callbacks
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=lgb_rmsle,
    callbacks=[
        early_stopping(stopping_rounds=10),
        log_evaluation(period=10)
    ]
)

# Predict
y_pred = model.predict(X_val)

# Evaluate
rmsle_final = rmsle(y_val, y_pred)
print(f'Mean Squared Error: {rmsle_final:.4f}')


# works, but not with the correct metric !
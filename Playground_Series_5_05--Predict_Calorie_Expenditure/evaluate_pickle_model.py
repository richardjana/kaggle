""" Load a model from *.pkl file, generate plots and make predictions to submit on kaggle.
"""
import sys

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from xgboost.sklearn import XGBRegressor

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import make_diagonal_plot, rmsle  # noqa
from prepare_calories_data import load_preprocess_data

TARGET_COL = 'Calories'
PKL_FILE = sys.argv[1]

# load model
model: LGBMRegressor | RandomForestRegressor | XGBRegressor = joblib.load(PKL_FILE)

# load dataset
log1p_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
train, test = load_preprocess_data('train.csv', 'test.csv', TARGET_COL)

y = train.pop(TARGET_COL).to_numpy()
X = train.to_numpy()

# predict training data
pred_train = model.predict(X)
pred_train = log1p_transformer.inverse_transform(pred_train.reshape(-1, 1))

make_diagonal_plot(pd.DataFrame({TARGET_COL: y.ravel(), 'PREDICTION': pred_train.ravel()}),
                        pd.DataFrame({TARGET_COL: y.ravel(), 'PREDICTION': pred_train.ravel()}),
                        TARGET_COL, rmsle, 'RMSLE',
                        'error_diagonal_from_pkl.png', precision=5)

# predict test data
test_preds = model.predict(test)
test_preds = log1p_transformer.inverse_transform(test_preds.reshape(-1, 1))

submit_df = pd.read_csv('sample_submission.csv')
submit_df[TARGET_COL] = test_preds
submit_df.to_csv('predictions_from_pkl.csv', columns=['id', TARGET_COL], index=False)

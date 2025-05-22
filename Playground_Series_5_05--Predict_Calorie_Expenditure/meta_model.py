from itertools import combinations
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import FunctionTransformer
import tensorflow as tf

sys.path.append('../')
from kaggle_utilities import make_diagonal_plot, rmsle  # noqa
from prepare_calories_data import load_preprocess_data

TARGET_COL = 'Calories'


# Load dataset
log1p_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
dataframe, test = load_preprocess_data('train.csv', 'test.csv', TARGET_COL, log1p_transformer)

y = dataframe.pop(TARGET_COL).to_numpy()
X = dataframe.to_numpy()

# Example: Split data
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Load pre-trained base models
base_models = [('rf', joblib.load('rf_model.pkl')),  # RandomForest
               ('xgb', joblib.load('xgb_model.pkl')),  # XGBoost
               ('lgb', joblib.load('lgb_model.pkl')),  # LightGBM
               ('tf', tf.keras.models.load_model('tf_model.h5'))  # TensorFlow
               ]

# Placeholder for meta features
meta_train = np.zeros((X_train_full.shape[0], len(base_models)))
meta_test = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    print(f"Using pre-trained base model: {name}")
    meta_test_fold = np.zeros((X_test.shape[0], kf.n_splits))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full)):
        X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]

        meta_train[val_idx, i] = model.predict(X_val)
        meta_test_fold[:, fold] = model.predict(X_test)

    # Average test predictions from each fold
    meta_test[:, i] = meta_test_fold.mean(axis=1)

# Train meta-model
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_train, y_train_full)

# Final predictions
final_pred = log1p_transformer.inverse_transform(meta_model.predict(meta_test))

# Evaluate
rmse_meta = root_mean_squared_error(y_test, final_pred)
print(f"Final RMSE (stacked model): {rmse_meta:.5f}")
rmsle_meta = rmsle(y_test, final_pred)
print(f"Final RMSLE (stacked model): {rmsle_meta:.5f}")

submit_df = pd.read_csv('sample_submission.csv')
submit_df[TARGET_COL] = final_pred
submit_df.to_csv('predictions_meta_model.csv', columns=['id', TARGET_COL], index=False)

make_diagonal_plot(pd.DataFrame({TARGET_COL: y_train_full, 'PREDICTION': final_pred}),
                    pd.DataFrame({TARGET_COL: y_train_full, 'PREDICTION': final_pred}),
                    TARGET_COL, rmsle, 'RMSLE',
                    'error_diagonal-meta_model.png', precision=5)

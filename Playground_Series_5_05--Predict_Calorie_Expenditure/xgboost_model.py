from typing import Dict, List
import sys

import joblib
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import FunctionTransformer

sys.path.append('../')
from kaggle_utilities import make_diagonal_plot, rmsle  # noqa
from prepare_calories_data import load_preprocess_data

TARGET_COL = 'Calories'

NUM_CV_SPLITS = 5

NUM_ESTIMATORS = 5000
LEARNING_RATE = 0.01
MAX_DEPTH = 4
SUBSAMPLE = 0.7
COLSAMPLE_BYTREE = 0.8
L1_REGULARIZATION = 0.1
L2_REGULARIZATION = 1.0


def make_training_plot(history: Dict[str, Dict[str, List[float]]], metric: str,
                       fname: str, precision: int = 2) -> None:
    """ Make plots to visualize the training progress. """
    _, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
    ax.plot(np.arange(len(history['validation_0'][metric]))+1, history['validation_0'][metric], 'r',
            label=f"training {metric} ({min(history['validation_0'][metric]):.{precision}f})")
    ax.plot(np.arange(len(history['validation_1'][metric]))+1, history['validation_1'][metric], 'g',
            label=f"validation {metric} ({min(history['validation_1'][metric]):.{precision}f})")
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    plt.legend(loc='best')

    ax.set_yscale('log')
    plt.savefig(f"{fname}_{metric}_LOG.png", bbox_inches='tight')

    plt.close()


def make_prediction(model: XGBRegressor, test_df: pd.DataFrame,
                    skl_transformer: FunctionTransformer, cv_index: int | str) -> None:
    """ Make a prediction for the test data. """
    submit_df = pd.read_csv('sample_submission.csv')
    submit_df[TARGET_COL] = model.predict(test_df.to_numpy())
    submit_df[TARGET_COL] = skl_transformer.inverse_transform(submit_df[[TARGET_COL]])
    submit_df.to_csv(f"predictions_XGBoost_KFold_{cv_index}.csv",
                     columns=['id', TARGET_COL], index=False)


# Load dataset
log1p_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
dataframe, test = load_preprocess_data('train.csv', 'test.csv', TARGET_COL, log1p_transformer)

y = dataframe.pop(TARGET_COL).to_numpy()
X = dataframe.to_numpy()

if NUM_CV_SPLITS > 1:  # do cross-validation
    kfold = KFold(n_splits=NUM_CV_SPLITS, shuffle=True, random_state=42)
    scores = []
    cv_index = 0

    for train_index, val_index in kfold.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_df = dataframe.iloc[train_index].copy()
        val_df = dataframe.iloc[val_index].copy()

        cv_index += 1

        # Create the model
        model = XGBRegressor(
            objective = 'reg:squarederror',
            n_estimators = NUM_ESTIMATORS,
            learning_rate = LEARNING_RATE,
            max_depth = MAX_DEPTH,
            subsample = SUBSAMPLE,
            colsample_bytree = COLSAMPLE_BYTREE,
            reg_alpha = L1_REGULARIZATION,
            reg_lambda = L2_REGULARIZATION,
            tree_method = 'auto',  # XGBoost typically works better with this
            verbosity = 1,
            early_stopping_rounds=100,
            eval_metric = 'rmse',
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50
        )

        eval_results = model.evals_result()

        make_training_plot(eval_results, 'rmse', f"training_XGBoost_{cv_index}", precision=5)

        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        y_train = log1p_transformer.inverse_transform(y_train)
        y_val = log1p_transformer.inverse_transform(y_val)
        pred_train = log1p_transformer.inverse_transform(pred_train)
        pred_val = log1p_transformer.inverse_transform(pred_val)

        make_diagonal_plot(pd.DataFrame({TARGET_COL: y_train, 'PREDICTION': pred_train}),
                        pd.DataFrame({TARGET_COL: y_val, 'PREDICTION': pred_val}),
                        TARGET_COL, rmsle, 'RMSLE',
                        f"error_diagonal_XGBoost_{cv_index}.png", precision=5)

        # Evaluate
        rmsle_final = rmsle(y_val, pred_val)
        print(f'Root Mean Squared Logarithmic Error: {rmsle_final:.7f}')

        # make prediction for the test data
        make_prediction(model, test, log1p_transformer, cv_index)

        scores.append(rmsle_final)

    print(f"Average cross-validation RMSLE: {np.mean(scores):.5f} ({scores})")

else:  # train on the full data set for final prediction
    cv_index = 'full'  # for plot names
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Create the model
    model = XGBRegressor(
        objective = 'reg:squarederror',
        n_estimators = NUM_ESTIMATORS,
        learning_rate = LEARNING_RATE,
        max_depth = MAX_DEPTH,
        subsample = SUBSAMPLE,
        colsample_bytree = COLSAMPLE_BYTREE,
        reg_alpha = L1_REGULARIZATION,
        reg_lambda = L2_REGULARIZATION,
        tree_method = 'auto',
        eval_metric = 'rmse',
        early_stopping_rounds=100,
        verbosity = 1
    )

    # Train with proper early stopping and logging callbacks
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )

    joblib.dump(model, 'xgb_model.pkl')

    eval_results = model.evals_result()

    make_training_plot(eval_results, 'rmse', 'training_XGBoost', precision=5)

    # Predict
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    y_train = log1p_transformer.inverse_transform(y_train)
    y_val = log1p_transformer.inverse_transform(y_val)
    pred_train = log1p_transformer.inverse_transform(pred_train)
    pred_val = log1p_transformer.inverse_transform(pred_val)

    make_diagonal_plot(pd.DataFrame({TARGET_COL: y_train, 'PREDICTION': pred_train}),
                    pd.DataFrame({TARGET_COL: y_val, 'PREDICTION': pred_val}),
                    TARGET_COL, rmsle, 'RMSLE',
                    f"error_diagonal_XGBoost_{cv_index}.png", precision=5)

    # Evaluate
    rmsle_final = rmsle(y_val, pred_val)
    print(f'Root Mean Squared Logarithmic Error: {rmsle_final:.7f}')

    # make prediction for the test data
    make_prediction(model, test, log1p_transformer, cv_index)

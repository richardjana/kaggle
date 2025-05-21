from itertools import combinations
from typing import Dict, List
import sys

import joblib
#import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import FunctionTransformer

sys.path.append('../')
from kaggle_utilities import make_diagonal_plot, rmsle  # noqa

TARGET_COL = 'Calories'

NUM_CV_SPLITS = 5

NUM_ESTIMATORS = 5000
LEARNING_RATE = 0.01
MAX_DEPTH = 6
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
L1_REGULARIZATION = 1.0
L2_REGULARIZATION = 1.0

def clean_data(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ Apply cleaning operations to pandas DataFrame. """
    pd_df.drop('id', axis=1, inplace=True)
    pd_df['Sex'] = pd_df['Sex'].map({'male': 0, 'female': 1})
    return pd_df

def add_intuitive_columns(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ Add intuitive columns to the DataFrame. """
    pd_df['BMI'] = pd_df['Weight'] / (pd_df['Height']**2) * 10000
    pd_df['BMI_Class'] = pd.cut(pd_df['BMI'],
                                bins=[0, 16.5, 18.5, 25, 30, 35, 40, 100],
                                labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)
    pd_df['BMI_zscore'] = pd_df.groupby('Sex')['BMI'].transform(zscore)

    BMR_male = 66.47 + pd_df['Weight']*13.75 + pd_df['Height']*5.003 - pd_df['Age']*6.755
    BMR_female = 655.1 + pd_df['Weight']*9.563 + pd_df['Height']*1.85 - pd_df['Age']*4.676
    pd_df['BMR'] = np.where(pd_df['Sex'] == 'male', BMR_male, BMR_female)
    pd_df['BMR_zscore'] = pd_df.groupby('Sex')['BMR'].transform(zscore)

    pd_df['Heart_rate_Zone'] = pd.cut(pd_df['Heart_Rate'], bins=[0, 90, 110, 200],
                                      labels=[0, 1, 2]).astype(int)
    pd_df['Heart_Rate_Zone_2'] = pd.cut(pd_df['Heart_Rate']/(220-pd_df['Age'])*100,
                                        bins=[0, 50, 65, 80, 85, 92, 100],
                                        labels=[0, 1, 2, 3, 4, 5]).astype(int)

    pd_df['Age_Group'] = pd.cut(pd_df['Age'], bins=[0, 20, 35, 50, 100],
                                labels=[0, 1, 2, 3]).astype(int)

    cb_male = (0.6309*pd_df['Heart_Rate'] + 0.1988*pd_df['Weight']
               + 0.2017*pd_df['Age'] - 55.0969) / 4.184 * pd_df['Duration']
    cb_female = (0.4472*pd_df['Heart_Rate'] - 0.1263*pd_df['Weight']
                 + 0.0740*pd_df['Age'] - 20.4022) / 4.184 * pd_df['Duration']
    pd_df['Calories_Burned'] = np.where(pd_df['Sex'] == 'male', cb_male, cb_female)

    for col in ['Height', 'Weight', 'Heart_Rate']:
        pd_df[f"{col}_zscore"] = pd_df.groupby('Sex')[col].transform(zscore)

    return pd_df


def generate_extra_columns(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ Generate extra feature columns from the original data. """
    combine_cols = [col for col in pd_df.keys() if col != TARGET_COL]
    new_cols = {}
    for n in [2, 3]:
        for cols in combinations(combine_cols, n):
            col_name = '*'.join(cols)
            new_cols[col_name] = pd_df[list(cols)].prod(axis=1)

    for cols in combinations(combine_cols, 2):
        col_name = '/'.join(cols)
        new_cols[col_name] = pd_df[cols[0]] / pd_df[cols[1]]

    pd_df = pd.concat([pd_df, pd.DataFrame(new_cols, index=pd_df.index)], axis=1)
    return pd_df

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
    plt.savefig(f"{fname}_{metric}.png", bbox_inches='tight')

    ax.set_yscale('log')
    plt.savefig(f"{fname}_{metric}_LOG.png", bbox_inches='tight')

    plt.close()

#def make_prediction(model: xgb.XGBRegressor, test_df: pd.DataFrame,
def make_prediction(model: XGBRegressor, test_df: pd.DataFrame,
                    skl_transformer: FunctionTransformer, cv_index: int | str) -> None:
    """ Make a prediction for the test data. """
    submit_df = pd.read_csv('sample_submission.csv')
    submit_df[TARGET_COL] = model.predict(test_df.to_numpy())
    submit_df[TARGET_COL] = skl_transformer.inverse_transform(submit_df[[TARGET_COL]])
    submit_df.to_csv(f"predictions_XGBoost_KFold_{cv_index}.csv",
                     columns=['id', TARGET_COL], index=False)


# Load dataset
dataframe = clean_data(pd.read_csv('train.csv'))
dataframe, rest = train_test_split(dataframe, test_size=0.80)  # reduce dataset size for testing
test = clean_data(pd.read_csv('test.csv'))

dataframe = generate_extra_columns(dataframe)
test = generate_extra_columns(test)

dataframe = add_intuitive_columns(dataframe)
test = add_intuitive_columns(test)

skl_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
dataframe[TARGET_COL] = skl_transformer.transform(dataframe[[TARGET_COL]])

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
        test_df = test.copy()

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

        y_train = skl_transformer.inverse_transform(y_train)
        y_val = skl_transformer.inverse_transform(y_val)
        pred_train = skl_transformer.inverse_transform(pred_train)
        pred_val = skl_transformer.inverse_transform(pred_val)

        make_diagonal_plot(pd.DataFrame({TARGET_COL: y_train, 'PREDICTION': pred_train}),
                        pd.DataFrame({TARGET_COL: y_val, 'PREDICTION': pred_val}),
                        TARGET_COL, rmsle, 'RMSLE',
                        f"error_diagonal_XGBoost_{cv_index}.png", precision=5)

        # Evaluate
        rmsle_final = rmsle(y_val, pred_val)
        print(f'Root Mean Squared Logarithmic Error: {rmsle_final:.7f}')

        # make prediction for the test data
        make_prediction(model, test, skl_transformer, cv_index)

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

    y_train = skl_transformer.inverse_transform(y_train)
    y_val = skl_transformer.inverse_transform(y_val)
    pred_train = skl_transformer.inverse_transform(pred_train)
    pred_val = skl_transformer.inverse_transform(pred_val)

    make_diagonal_plot(pd.DataFrame({TARGET_COL: y_train, 'PREDICTION': pred_train}),
                    pd.DataFrame({TARGET_COL: y_val, 'PREDICTION': pred_val}),
                    TARGET_COL, rmsle, 'RMSLE',
                    f"error_diagonal_XGBoost_{cv_index}.png", precision=5)

    # Evaluate
    rmsle_final = rmsle(y_val, pred_val)
    print(f'Root Mean Squared Logarithmic Error: {rmsle_final:.7f}')

    # make prediction for the test data
    make_prediction(model, test, skl_transformer, cv_index)

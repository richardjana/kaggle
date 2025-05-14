from itertools import combinations
import sys
from typing import Dict, List, Literal, Tuple, Union

import keras
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf

sys.path.append('../')
from kaggle_utilities import min_max_scaler, make_training_plot, make_diagonal_plot, rmsle, rmsle_metric  # noqa

##### hyper params for the model #####
LAYER_SIZE = 64
L2_REG = 0.01 / 10
DROP_RATE = 0.25
LEARNING_RATE_INITIAL = 1e-4
LEARNING_RATE_FINAL = 1e-6
NUM_EPOCHS = 500
BATCH_SIZE = 128
NUM_CV_SPLITS = 1
TARGET_COL = 'Calories'

LOSS_FUNCTION = tf.keras.losses.MeanSquaredLogarithmicError()
METRIC = rmsle_metric
#LOSS_FUNCTION = tf.keras.losses.MeanSquaredError()
#METRIC = 'root_mean_squared_error'

from sklearn.base import BaseEstimator, TransformerMixin
class PositivePowerTransformer(BaseEstimator, TransformerMixin):
    """ Modify the sklearn PowerTransformer by shifting the transformed values. This is done in
        order to obtain exclusively positive values for the RMSLE metric.
    """
    def __init__(self, method: Literal['yeo-johnson', 'box-cox'] = 'yeo-johnson') -> None:
        self.transformer: PowerTransformer = PowerTransformer(method=method)
        self.shift_value: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'PositivePowerTransformer':
        """ Fit the transformer to the data."""
        self.transformer.fit(X)

        X_transformed: np.ndarray = self.transformer.transform(X)

        X_min: float = X_transformed.min()
        if X_min < 0:
            self.shift_value = abs(X_min)
        else:
            self.shift_value = 0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """ Transform the data, ensuring no negative values."""
        return self.transformer.transform(X) + self.shift_value

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """ Inverse transform the data."""
        return self.transformer.inverse_transform(X - self.shift_value)

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

# probably most important: Duration >> Body_Temp > Heart_Rate >> rest

def add_intuitive_columns(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ Add columns to DataFrame, possibly inspired by other peoples solutions.
    Args:
        pd_df (pd.DataFrame): DataFrame to which to add new columns.
    Returns:
        pd.DataFrame: The DataFrame, with additional columns.
    """
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

    return pd_df


def generate_extra_columns(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ Generate extra feature columns from the original data, by combining columns.
    Args:
        pd_df (pd.DataFrame): The original data.
    Returns:
        pd.DataFrame: DataFrame with new columns added.
    """
    combine_cols = [col for col in pd_df.keys() if col != TARGET_COL]

    new_cols = {}

    for n in [2]:
        for cols in combinations(combine_cols, n):
            col_name = '*'.join(cols)
            new_cols[col_name] = pd_df[list(cols)].prod(axis=1)

    for cols in combinations(combine_cols, 2):
        col_name = '/'.join(cols)
        new_cols[col_name] = pd_df[cols[0]] / pd_df[cols[1]]

    pd_df = pd.concat([pd_df, pd.DataFrame(new_cols, index=pd_df.index)], axis=1)

    return pd_df


##### load data #####
dataframe = clean_data(pd.read_csv('train.csv'))
# reduce dataset size for testing
#dataframe, rest = train_test_split(dataframe, test_size=0.95)
test = clean_data(pd.read_csv('test.csv'))

dataframe = generate_extra_columns(dataframe)
test = generate_extra_columns(test)

dataframe = add_intuitive_columns(dataframe)
test = add_intuitive_columns(test)


def make_new_model(shape: int) -> tf.keras.Model:
    """ Create a fresh model, with fresh weights and clean optimizer state.
    Args:
        shape (int): Shape (size) of the input layer.
    Returns:
        tf.keras.Model: The compiled model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(shape,)),
        tf.keras.layers.Dense(LAYER_SIZE, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
        tf.keras.layers.Dropout(DROP_RATE),
        tf.keras.layers.Dense(LAYER_SIZE, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
        tf.keras.layers.Dropout(DROP_RATE),
        tf.keras.layers.Dense(LAYER_SIZE, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
        tf.keras.layers.Dropout(DROP_RATE),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=LOSS_FUNCTION,
                  metrics=[METRIC])

    return model


def make_prediction(model: tf.keras.Model, test_df_encoded: pd.DataFrame,
                    skl_pt: PowerTransformer, cv_index: int | str) -> None:
    """ Make a prediction for the test data, with a given model.
    Args:
        model (tf.keras.Model): Model used for the prediction.
        test_df_encoded (pd.DataFrame): DataFrame with the test data, pre-processed.
        skl_pt (PowerTransformer): _description_
        cv_index (Union[int, str]): Index of the cross-validation fold, used in the file name.
    """
    submit_df = pd.read_csv('sample_submission.csv')
    submit_df[TARGET_COL] = model.predict(test_df_encoded.to_numpy())
    #submit_df[TARGET_COL] = skl_pt.inverse_transform(submit_df[[TARGET_COL]])
    submit_df.to_csv(f"predictions_KFold_{cv_index}.csv", columns=['id', TARGET_COL], index=False)


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=round(NUM_EPOCHS**(1/2)),
    restore_best_weights=True
)


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                test_df: pd.DataFrame, cv_index: int | str) -> float:
    """ min-max-scale columns, train the model and make predictions and plots.
    Args:
        train_df (pd.DataFrame): _description_
        val_df (pd.DataFrame): _description_
        test_df (pd.DataFrame): _description_
        cv_index (int): _description_
    Returns:
        float: Score for the model between training and validation data.
    """
    # defragment DataFrames
    train_df._consolidate_inplace()
    val_df._consolidate_inplace()
    test_df._consolidate_inplace()

    # scale columns
    scale_columns = [col for col in train_df.keys() if col != TARGET_COL]
    train_df, val_df, test_df = min_max_scaler([train_df, val_df, test_df], scale_columns)

    skl_pt = PositivePowerTransformer(method='yeo-johnson')
    #train_df[TARGET_COL] = skl_pt.fit_transform(train_df[[TARGET_COL]])
    #val_df[TARGET_COL] = skl_pt.transform(val_df[[TARGET_COL]])

    train_target_df = pd.DataFrame({TARGET_COL: train_df.pop(TARGET_COL)})
    y_train = train_target_df.to_numpy()
    X_train = train_df.to_numpy()
    val_target_df = pd.DataFrame({TARGET_COL: val_df.pop(TARGET_COL)})
    y_val = val_target_df.to_numpy()
    X_val = val_df.to_numpy()

    model = make_new_model(shape=X_train.shape[1])
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[early_stop])

    make_training_plot(history.history, f"training_KFold_{cv_index}.png", precision=5)

    train_target_df['PREDICTION'] = model.predict(X_train)
    val_target_df['PREDICTION'] = model.predict(X_val)

    make_diagonal_plot(train_target_df, val_target_df, TARGET_COL, rmsle,
                       'RMSLE', f"error_diagonal_{cv_index}.png", precision=5)

    make_prediction(model, test_df, skl_pt, cv_index)

    score = rmsle(val_target_df[TARGET_COL].to_numpy(),
                  val_target_df['PREDICTION'].to_numpy())

    return score


if NUM_CV_SPLITS > 1:  # do cross-validation
    kfold = KFold(n_splits=NUM_CV_SPLITS, shuffle=True)
    scores = []
    cv_index = 0

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE_INITIAL,
        decay_steps=len(dataframe) // NUM_CV_SPLITS *
        (NUM_CV_SPLITS-1) // BATCH_SIZE,
        decay_rate=(LEARNING_RATE_FINAL /
                    LEARNING_RATE_INITIAL) ** (1 / (NUM_EPOCHS - 1)),
        staircase=True
    )

    for train_index, val_index in kfold.split(dataframe):
        train_df = dataframe.iloc[train_index].copy()
        val_df = dataframe.iloc[val_index].copy()
        test_df = test.copy()

        cv_index += 1
        cv_split_score = train_model(train_df, val_df, test_df, cv_index)
        scores.append(cv_split_score)

    print(f'Average cross-validation RMSLE: {np.mean(scores):.4f} ({scores})')

else:  # train on the full data set for final prediction
    cv_index = 'full'  # for plot names

    train_df, val_df = train_test_split(dataframe, test_size=0.1)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE_INITIAL,
        decay_steps=len(dataframe) // BATCH_SIZE,
        decay_rate=(LEARNING_RATE_FINAL /
                    LEARNING_RATE_INITIAL) ** (1 / (NUM_EPOCHS - 1)),
        staircase=True
    )

    score = train_model(train_df, val_df, test, cv_index)

    print(f"Final RMSLE score = {score}")

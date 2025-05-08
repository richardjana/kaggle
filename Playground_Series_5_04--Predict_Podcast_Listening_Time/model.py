from itertools import combinations
import sys
from typing import Dict, List, Tuple

from collections import Counter
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import TargetEncoder
import tensorflow as tf

sys.path.append('../')
from kaggle_utilities import min_max_scaler, make_training_plot, make_diagonal_plot, RMSE  # noqa

##### hyper params for the model #####
LAYER_SIZE = 64
L2_REG = 0.01 / 10
DROP_RATE = 0.25
LEARNING_RATE_INITIAL = 1e-4
LEARNING_RATE_FINAL = 1e-6
NUM_EPOCHS = 1000
BATCH_SIZE = 128
NUM_CV_SPLITS = 5
TARGET_COL = 'Listening_Time_minutes'

LOSS_FUNCTION = tf.keras.losses.MeanSquaredError()
METRIC = 'root_mean_squared_error'


def clean_data(pd_df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
    """ Apply cleaning operations to pandas DataFrame.
    Args:
        pd_df (pd.DataFrame): The DataFrame to clean.
        drop (bool, optional): If rows should be dropped (option for training DataFrame) or not
            (test DataFrame). Defaults to True.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    pd_df.drop('id', axis=1, inplace=True)

    if drop is True:  # fix erroneous values / data points ...
        pd_df.drop(pd_df[pd_df['Episode_Length_minutes']
                   > 180].index, inplace=True)
        pd_df.drop(pd_df[pd_df['Number_of_Ads'] > 10].index, inplace=True)
    else:
        pd_df.loc[pd_df['Episode_Length_minutes'] > 180,
                  'Episode_Length_minutes'] = pd_df['Episode_Length_minutes'].median()
        pd_df.loc[pd_df['Number_of_Ads'] > 10,
                  'Number_of_Ads'] = pd_df['Number_of_Ads'].median()

    if drop is True:  # drop NaN lines
        pd_df.dropna(axis=0, how='any', inplace=True)
    else:  # for the test set, fill with most common / average
        for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']:
            pd_df[col] = pd_df[col].fillna(pd_df[col].median())

    # make 'Episode_Title' integer column (all follow 'Episode_<number> pattern) ...
    pd_df['Episode_Title'] = pd_df['Episode_Title'].map(
        lambda et: int(et.split()[1]))
    # ... or drop it altogether?
    # pd_df.drop('Episode_Title', axis=1, inplace=True)

    if drop is True:  # drop / replace outliers / implausible data points
        pd_df.drop(pd_df[pd_df['Host_Popularity_percentage']
                   < 20].index, inplace=True)
        pd_df.drop(pd_df[pd_df['Host_Popularity_percentage']
                   > 100].index, inplace=True)
        pd_df.drop(pd_df[pd_df['Guest_Popularity_percentage']
                   < 0].index, inplace=True)
        pd_df.drop(pd_df[pd_df['Guest_Popularity_percentage']
                   > 100].index, inplace=True)
    else:
        col = 'Host_Popularity_percentage'  # 20 <= Host_Popularity_percentage <= 100
        valid_median = pd_df[col][(pd_df[col] >= 20)
                                  & (pd_df[col] <= 100)].median()
        pd_df.loc[pd_df[(pd_df[col] < 20) | (pd_df[col] > 100)
                        ].index, col] = valid_median
        col = 'Guest_Popularity_percentage'  # 0 <= Guest_Popularity_percentage <= 100
        valid_median = pd_df[col][(pd_df[col] >= 0)
                                  & (pd_df[col] <= 100)].median()
        pd_df.loc[pd_df[(pd_df[col] < 0) | (pd_df[col] > 100)
                        ].index, col] = valid_median

    return pd_df


def add_intuitive_columns(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ Add columns to DataFrame, inspired by other peoples solutions.
    Args:
        pd_df (pd.DataFrame): DataFrame to which to add new columns.
    Returns:
        pd.DataFrame: The DataFrame, with additional columns.
    """
    pd_df['Ads_per_Minute'] = pd_df['Number_of_Ads'] / \
        pd_df['Episode_Length_minutes']
    pd_df['Popularity_sum'] = pd_df['Host_Popularity_percentage'] + \
        pd_df['Guest_Popularity_percentage']
    pd_df['Popularity_diff'] = (
        pd_df['Host_Popularity_percentage'] - pd_df['Guest_Popularity_percentage']).abs()

    return pd_df


def generate_extra_columns(df_train: pd.DataFrame,
                           df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Generate additional columns from the raw data.
    Args:
        df_train (pd.DataFrame): The training DataFrame.
        df_test (pd.DataFrame): The test Dataframe.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and test DataFrames with added columns.
    """
    # column pairs, inspired by greysky
    encode_columns = [col for col in df_train.keys() if col != TARGET_COL]
    pair_size = [2, 3, 4]

    train_new_cols = {}
    test_new_cols = {}

    for r in pair_size:
        for cols in list(combinations(encode_columns, r)):
            new_col_name = '_'.join(cols)

            train_new_cols[new_col_name] = df_train[list(cols)].astype(
                str).agg('_'.join, axis=1).astype('category')
            test_new_cols[new_col_name] = df_test[list(cols)].astype(
                str).agg('_'.join, axis=1).astype('category')

    df_train = pd.concat([df_train, pd.DataFrame(
        train_new_cols, index=df_train.index)], axis=1)
    df_test = pd.concat([df_test, pd.DataFrame(
        test_new_cols, index=df_test.index)], axis=1)

    return (df_train, df_test)


##### load data #####
dataframe = clean_data(pd.read_csv('train.csv'))
# reduce dataset size for testing
# dataframe, rest = train_test_split(dataframe, test_size=0.999)
test = clean_data(pd.read_csv('test.csv'), drop=False)

dataframe, test = generate_extra_columns(dataframe, test)

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


def make_prediction(model: tf.keras.Model, test_df_encoded: pd.DataFrame, cv_index: int) -> None:
    """ Make a prediction for the test data, with a given model.
    Args:
        model (_type_): Model used for the prediction.
        i (int): Index of the cross-validation fold, used in the file name.
    """
    submit_df = pd.read_csv('sample_submission.csv')
    submit_df[TARGET_COL] = model.predict(test_df_encoded.to_numpy())
    submit_df.to_csv(f"predictions_KFold_{cv_index}.csv", columns=[
        'id', TARGET_COL], index=False)


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=round(NUM_EPOCHS**(1/3)),
    restore_best_weights=True
)


def train_model(train_df, val_df, test_df, cv_index):
    """ Target encode category columns, min-max-scale, train the model and make predictions and plots.

    Args:
        train_df (_type_): _description_
        val_df (_type_): _description_
        test_df (_type_): _description_
        cv_index (_type_): _description_

    Returns:
        _type_: _description_
    """
    # encode category columns
    encoded_columns = train_df.select_dtypes(
        include=['object', 'category']).columns.tolist()
    encoder = TargetEncoder(random_state=42)

    train_df[encoded_columns] = encoder.fit_transform(
        train_df[encoded_columns], train_df[TARGET_COL])

    val_df[encoded_columns] = encoder.transform(val_df[encoded_columns])
    test_df[encoded_columns] = encoder.transform(test_df[encoded_columns])

    # defragment DataFrames
    train_df._consolidate_inplace()
    val_df._consolidate_inplace()
    test_df._consolidate_inplace()

    # scale columns
    scale_columns = [col for col in train_df.keys() if col != TARGET_COL]
    train_df, val_df, test_df = min_max_scaler(
        [train_df, val_df, test_df], scale_columns)

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

    model.save(f"podcast_KFold_{cv_index}.keras")
    make_training_plot(history.history, f"training_KFold_{cv_index}.png")

    train_target_df['PREDICTION'] = model.predict(X_train)
    val_target_df['PREDICTION'] = model.predict(X_val)

    make_diagonal_plot(train_target_df, val_target_df, TARGET_COL, RMSE,
                       'RMSE', f"error_diagonal_{cv_index}.png")

    make_prediction(model, test_df, cv_index)

    score = RMSE(val_target_df[TARGET_COL].to_numpy(),
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

    print(f'Average cross-validation RMSE: {np.mean(scores):.4f} ({scores})')

    # ensemble prediction over CV folds
    prediction_dfs = [pd.read_csv(
        f"predictions_KFold_{i}.csv") for i in range(NUM_CV_SPLITS)]
    pred_matrix = np.stack(
        [df[TARGET_COL].values for df in prediction_dfs], axis=1)
    ensemble_prediction = prediction_dfs[0][['id']].copy()
    ensemble_prediction[TARGET_COL] = pred_matrix.mean(axis=1)
    ensemble_prediction.to_csv('predictions_KFold_ensemble.csv', index=False)

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

    print(f"Final RMSE score = {score}")

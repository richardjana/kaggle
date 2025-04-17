from itertools import combinations
import sys
from typing import Dict, List, Tuple

from collections import Counter
import keras
import numpy as np
import pandas as pd
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf

sys.path.append('../')
from kaggle_utilities import min_max_scaler, make_training_plot, make_diagonal_plot, RMSE  # noqa

##### hyper params for the model #####
LAYER_SIZE = 64
L2_REG = 0.01 / 10
DROP_RATE = 0.25
LEARNING_RATE = 0.00001
NUM_EPOCHS = 1000
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
    # pd_df['Episode_Title'] = pd_df['Episode_Title'].map(lambda et: int(et.split()[1]))
    # ... or drop it altogether?
    pd_df.drop('Episode_Title', axis=1, inplace=True)

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
    encode_columns = ['Podcast_Name', 'Episode_Length_minutes', 'Genre',
                      'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time',
                      'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment']
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


def target_encode(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                  target_col: str, col_list: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Target encode all columns listed in col_list, both in training and test DataFrames.
        (Using the mean of the target column from the training DataFrame.)
    Args:
        df_train (pd.DataFrame): Training DataFrame.
        df_val (pd.DataFrame): Validation DataFrame.
        df_test (pd.DataFrame): Test DataFrame.
        target_col (str): Name of the target column to use for the encoding.
        col_list (List[str]): List of column names to target encode.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation and test
        DataFrames with encoded columns.
    """
    train_enc = {}
    val_enc = {}
    test_enc = {}

    for col in col_list:
        groupby_df = df_train[[target_col, col]].groupby(
            col, observed=True).mean()
        mapping_dict = groupby_df[target_col].to_dict()

        train_enc[col] = df_train[col].map(mapping_dict).astype(float)
        val_enc[col] = df_val[col].map(mapping_dict).astype(float)
        test_enc[col] = df_test[col].map(mapping_dict).astype(float)

    df_train[col_list] = pd.DataFrame(train_enc, index=df_train.index)
    df_val[col_list] = pd.DataFrame(val_enc, index=df_val.index)
    df_test[col_list] = pd.DataFrame(test_enc, index=df_test.index)

    return (df_train, df_val, df_test)


def count_encode(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                 col_list: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Encode categorical columns by the count of the unique labels in each. (Sum of train
        and test DataFrames.)
    Args:
        df_train (pd.DataFrame): Training DataFrame.
        df_val (pd.DataFrame): Validation DataFrame.
        df_test (pd.DataFrame): Test DataFrame.
        col_list (List[str]): List of column names to count encode.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation and test
        DataFrames with encoded columns.
    """

    def check_duplicate_counts(mapping_dict: Dict) -> bool:
        """ Check the mapping dictionary for collisions.
        Args:
            mapping_dict (Dict): Dictionary for the count encoding.
        Returns:
            bool: Are there collisions?
        """
        values = list(mapping_dict.values())
        return len(values) != len(set(values))

    for col in col_list:
        mapping_dict_train = df_train[col].value_counts().to_dict()
        mapping_dict_val = df_val[col].value_counts().to_dict()
        mapping_dict_test = df_test[col].value_counts().to_dict()

        mapping_dict = dict(Counter(mapping_dict_train) +
                            Counter(mapping_dict_val) +
                            Counter(mapping_dict_test))

        if check_duplicate_counts(mapping_dict):
            print(f"Duplicate counts in column '{col}'!")
        df_train[col] = df_train[col].map(mapping_dict).astype(int)
        df_val[col] = df_val[col].map(mapping_dict).astype(int)
        df_test[col] = df_test[col].map(mapping_dict).astype(int)

    return (df_train, df_val, df_test)


##### load data #####
dataframe = clean_data(pd.read_csv('train.csv'))
# reduce dataset size for testing
# dataframe, rest = train_test_split(dataframe, test_size=0.999)
test = clean_data(pd.read_csv('test.csv'), drop=False)

dataframe = add_intuitive_columns(dataframe)
test = add_intuitive_columns(test)

dataframe, test = generate_extra_columns(dataframe, test)


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

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
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


kfold = KFold(n_splits=NUM_CV_SPLITS, shuffle=True)
scores = []

cv_index = 0
for train_index, val_index in kfold.split(dataframe):
    train_df = dataframe.iloc[train_index].copy()
    val_df = dataframe.iloc[val_index].copy()

    # encode category columns for the fold
    category_columns = dataframe.select_dtypes(
        include=['category', 'object']).columns.tolist()
    train_df_enc, val_df_enc, test_df_enc = target_encode(
        train_df, val_df, test, TARGET_COL, category_columns)
    # train_df_enc, val_df_enc, test_df_enc = count_encode(
    #   train_df, val_df, test, category_columns)

    # defragment DataFrames
    train_df_enc._consolidate_inplace()
    val_df_enc._consolidate_inplace()
    test_df_enc._consolidate_inplace()

    ### scale columns (not cyclical representations, not target column) ###
    scale_columns = [col for col in dataframe.keys() if col != TARGET_COL]
    train_df_enc, val_df_enc, test_df_enc = min_max_scaler(
        [train_df_enc, val_df_enc, test_df_enc], scale_columns)

    train_target_df = pd.DataFrame({TARGET_COL: train_df_enc.pop(TARGET_COL)})
    y_train = train_target_df.to_numpy()
    X_train = train_df_enc.to_numpy()
    val_target_df = pd.DataFrame({TARGET_COL: val_df_enc.pop(TARGET_COL)})
    y_val = val_target_df.to_numpy()
    X_val = val_df_enc.to_numpy()

    model = make_new_model(shape=X_train.shape[1])
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=NUM_EPOCHS)

    model.save(f"rainfall_KFold_{cv_index}.keras")
    make_training_plot(history.history, f"training_KFold_{cv_index}.png")

    train_target_df['PREDICTION'] = model.predict(X_train)
    val_target_df['PREDICTION'] = model.predict(X_val)

    make_diagonal_plot(train_target_df, val_target_df, TARGET_COL, RMSE,
                       'RMSE', f"error_diagonal_{cv_index}.png")

    make_prediction(model, test_df_enc, cv_index)

    cv_index += 1
    scores.append(history.history[f"val_{METRIC}"][-1])

print(f'Average cross-validation RMSE: {np.mean(scores):.4f} ({scores})')

from itertools import combinations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


TARGET_COL = 'Fertilizer Name'
COMPETITION_NAME = 'playground-series-s5e6'

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
    pd_df.rename(columns={'Temparature': 'Temperature'}, inplace=True)

    return pd_df


def add_fertilizer_components(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ Translate the fertilizer names into percentages of the chemical components.
    Args:
        pd_df (pd.DataFrame): DataFrame to which to add new columns.
    Returns:
        pd.DataFrame: The DataFrame, with additional columns.
    """
    fertilizer_compositions = {'28-28': {'N': 28, 'P2O5': 28, 'K2O': 0},
                               '17-17-17': {'N': 17, 'P2O5': 17, 'K2O': 17},
                               '10-26-26': {'N': 10, 'P2O5': 26, 'K2O': 26},
                               'DAP': {'N': 18, 'P2O5': 46, 'K2O': 0},
                               '20-20': {'N': 20, 'P2O5': 20, 'K2O': 0},
                               '14-35-14': {'N': 14, 'P2O5': 35, 'K2O': 14},
                               'Urea': {'N': 46, 'P2O5': 0, 'K2O': 0}
                               }

    composition_df = pd.DataFrame.from_dict(fertilizer_compositions, orient='index').reset_index()
    composition_df.rename(columns={'index': 'Fertilizer Name'}, inplace=True)

    return pd_df.merge(composition_df, on='Fertilizer Name', how='left')


def encode_category_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, framework: str
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    categorical_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']

    if framework in ['LGBM', 'XGB']:
        label_encoders = {}
        for col in categorical_cols:
            try:
                le = LabelEncoder()
                train_df[col] = le.fit_transform(train_df[col])
                test_df[col] = le.transform(test_df[col])
            except:
                continue
            label_encoders[col] = le

        return train_df, test_df, label_encoders

    else:
        return train_df, test_df, {}


def target_encode_multi_class(train_df: pd.DataFrame, test_df: pd.DataFrame, encode_col: str,
                              target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Target encode a column for multi-class classification problem. Creates N-1 new columns
        with the probabilities for the target labels and removes 'encode_col'.
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        encode_col (str): Column to encode.
        target_col (str): Target column used to encode.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The encoded training and test DataFrames.
    """
    counts_df = train_df.groupby(encode_col)[target_col].value_counts().unstack(fill_value=0)
    counts_df = counts_df.div(counts_df.sum(axis=1), axis=0)

    train_df = pd.merge(train_df, counts_df, on=encode_col, how='left').fillna(0)
    train_df.drop(columns=[encode_col, train_df[TARGET_COL].iloc[0]])
    test_df = pd.merge(train_df, counts_df, on=encode_col, how='left').fillna(0)
    test_df.drop(columns=[encode_col, train_df[TARGET_COL].iloc[0]])

    return train_df, test_df


def target_encode_multi_class_stratified(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                         encode_col: str, target_col: str, n_folds: int = 5
                                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Target encode a column for multi-class classification problem. Creates N-1 new columns
        with the probabilities for the target labels and removes 'encode_col'. The test set is
        encoded using the full training set. The training set is encoded using a stratified
        k-fold approach.
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        encode_col (str): Column to encode.
        target_col (str): Target column used to encode.
        n_folds (int, optional): Number of folds for encoding the training data. Defaults to 5.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The encoded training and test DataFrames.
    """
    def compute_target_encoding(df: pd.DataFrame, encode_col: str, target_col: str) -> pd.DataFrame:
        """ Returns a DataFrame with the normalized class proportions for each category in
            'encode_col'.
        Args:
            df (pd.DataFrame): Data to encode.
            encode_col (str): Column to encode.
            target_col (str): Target column used to encode.
        Returns:
            pd.DataFrame: DataFrame with the encoding values.
        """
        counts = df.groupby(encode_col)[target_col].value_counts().unstack(fill_value=0)
        proportions = counts.div(counts.sum(axis=1), axis=0)
        return proportions

    def apply_target_encoding(df: pd.DataFrame, encode_col: str, encoding_table: pd.DataFrame
                              ) -> pd.DataFrame:
        """ Apply precomputed encoding_table to DataFrame and return a new DataFrame.
        Args:
            df (pd.DataFrame): Data to encode.
            encode_col (str): Column to encode (join on).
            encoding_table (pd.DataFrame): Encoding values to join.
        Returns:
            pd.DataFrame: Encoded DataFrame, without the 'encode_col'.
        """
        encoded = df[[encode_col]].merge(encoding_table, on=encode_col, how='left').fillna(0)

        return encoded.drop(columns=[encode_col])

    encoded_cols = train_df[target_col].unique()

    # encode test_df with full train_df
    encoding_table = compute_target_encoding(train_df, encode_col, target_col)
    test_encoded = apply_target_encoding(test_df, encode_col, encoding_table)
    test_df[encoded_cols] = test_encoded.astype(float)
    test_df = test_df.drop(columns=[encode_col, encoded_cols[0]])

    # encode train_df, stratified k-fold
    train_encoded = pd.DataFrame(index=train_df.index, columns=encoded_cols)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for _, val_idx in kf.split(train_df):
        fold_val = train_df.iloc[val_idx]

        encoding_table = compute_target_encoding(fold_val, encode_col, target_col)
        fold_val_encoded = apply_target_encoding(fold_val, encode_col, encoding_table)
        train_encoded.iloc[val_idx] = fold_val_encoded.values

    train_df[encoded_cols] = train_encoded.astype(float)
    train_df = train_df.drop(columns=[encode_col, encoded_cols[0]])

    return train_df, test_df


def add_nutrient_chemistry(df: pd.DataFrame) -> pd.DataFrame:
    #N/P, N/K, P/K, plus total N+P+K
    df['N/P'] = df['Nitrogen'] / df['Phosphorous']
    df['N/K'] = df['Nitrogen'] / df['Potassium']
    df['P/K'] = df['Phosphorous'] / df['Potassium']
    df['N+P+K'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']

    return df


def load_preprocess_data(framework: str,
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """ Prepare training and test data into pandas DataFrames: added columns, transformations, etc.
    Args:
        framework (str): Name of the used framework, to encode categories correctly.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and test data, ready for model training.
    """
    train = clean_data(pd.read_csv('train.csv'))
    original = clean_data(pd.read_csv('Fertilizer_Prediction.csv'))
    train = pd.concat([train, original], ignore_index=True)
    #from sklearn.model_selection import train_test_split
    #train, _ = train_test_split(train, test_size=0.9)
    test = clean_data(pd.read_csv('test.csv'))

    train['sc-interaction'] = train['Soil Type'].str.cat(train['Crop Type'], sep=' ')
    train.drop(columns=['Soil Type', 'Crop Type'])
    test['sc-interaction'] = test['Soil Type'].str.cat(test['Crop Type'], sep=' ')
    test.drop(columns=['Soil Type', 'Crop Type'])
    #train, test = target_encode_multi_class(train, test, 'sc-interaction', TARGET_COL)
    train, test = target_encode_multi_class_stratified(train, test, 'sc-interaction', TARGET_COL)

    train, test, encoders = encode_category_columns(train, test, framework)

    #train = generate_extra_columns(train, target_col)
    #test = generate_extra_columns(test, target_col)

    #train = add_intuitive_columns(train)
    #test = add_intuitive_columns(test)

    return train, test, encoders

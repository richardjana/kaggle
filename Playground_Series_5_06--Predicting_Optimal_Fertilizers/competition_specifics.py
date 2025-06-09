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
    try:
        pd_df.drop('id', axis=1, inplace=True)
    except KeyError:
        pass
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
                label_encoders[col] = le
            except:
                continue
            try:
                test_df[col] = le.transform(test_df[col])
            except:
                pass

        return train_df, test_df, label_encoders

    else:
        return train_df, test_df, {}


def target_encode_multi_class(df: pd.DataFrame, df_average: pd.DataFrame, encode_col: str,
                              target_col: str) -> pd.DataFrame:
    """ Target encode a column for multi-class classification problem. Creates N-1 new columns
        with the probabilities for the target labels and removes 'encode_col'.
    Args:
        df (pd.DataFrame): DataFrame to encode.
        df_average (pd.DataFrame): Data used to calculate the encoding values. (In CV, the same
            as df, for test the full training df.)
        encode_col (str): Name of the column to encode.
        target_col (str): Name of the target column.
    Returns:
        pd.DataFrame: DataFrame with the new columns added and the encoded column dropped.
    """
    # compute target encoding
    counts = df_average.groupby(encode_col)[target_col].value_counts().unstack(fill_value=0)
    encoding_table = counts.div(counts.sum(axis=1), axis=0)

    # apply encoding
    df_encoded = df.merge(encoding_table, on=encode_col, how='left').fillna(0) # [[encode_col]]

    return df_encoded.drop(columns=[encode_col])


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
    train.drop(columns=['Soil Type', 'Crop Type'], inplace=True)  # optional
    test['sc-interaction'] = test['Soil Type'].str.cat(test['Crop Type'], sep=' ')
    test.drop(columns=['Soil Type', 'Crop Type'], inplace=True)  # optional

    train, test, encoders = encode_category_columns(train, test, framework)

    #train = generate_extra_columns(train, target_col)
    #test = generate_extra_columns(test, target_col)

    #train = add_intuitive_columns(train)
    #test = add_intuitive_columns(test)

    return train, test, encoders

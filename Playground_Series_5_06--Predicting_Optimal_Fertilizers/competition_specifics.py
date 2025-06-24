from itertools import combinations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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


def encode_category_columns(train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            original_df: pd.DataFrame,
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """ Encode categorical columns: label encoding for the TARGET_COL, one-hot encoding for all
        other categories.
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        original_df (pd.DataFrame): The original dataset.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]: Encoded training data, test data, original
         data and label encoder to inverse transform the target.
    """
    le = LabelEncoder()
    train_df[TARGET_COL] = le.fit_transform(train_df[TARGET_COL])
    original_df[TARGET_COL] = le.transform(original_df[TARGET_COL])

    combined = pd.concat([train_df, test_df, original_df], axis=0)  # for consistent encoding

    categorical_cols = ['Soil Type', 'Crop Type']
    available_cols = [col for col in categorical_cols if col in train_df.columns]

    combined_encoded = pd.get_dummies(combined, columns=available_cols)

    train_df_encoded = combined_encoded.iloc[:len(train_df), :]
    test_df_encoded = combined_encoded.iloc[len(train_df):-len(original_df), :]
    test_df_encoded.drop(columns=[TARGET_COL], inplace=True)
    original_df_encoded = combined_encoded.iloc[-len(original_df):, :]

    return train_df_encoded, test_df_encoded, original_df_encoded, le


def add_nutrient_chemistry(df: pd.DataFrame) -> pd.DataFrame:
    #N/P, N/K, P/K, plus total N+P+K
    df['N/P'] = df['Nitrogen'] / df['Phosphorous']
    df['N/K'] = df['Nitrogen'] / df['Potassium']
    df['P/K'] = df['Phosphorous'] / df['Potassium']
    df['N+P+K'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']

    df['Humidity_Temp_ratio'] = df['Humidity'] / df['Temperature']
    df['Moisture_Temp_ratio'] = df['Moisture'] / df['Temperature']

    return df


def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    """ Engineered features, as recommended by Komil Parmar.
    Args:
        df (pd.DataFrame): A DataFrame ...
    Returns:
        pd.DataFrame: ... with the added columns.
    """
    def is_temp_suitable(row: pd.Series) -> int:
        crop_temp_map = {'Barley': (15, 25),
                         'Cotton': (25, 35),
                         'Ground Nuts': (25, 32),
                         'Maize': (25, 32),
                         'Millets': (25, 35),
                         'Oil seeds': (20, 30),
                         'Paddy': (25, 35),
                         'Pulses': (20, 30),
                         'Sugarcane': (26, 35),
                         'Tobacco': (20, 30),
                         'Wheat': (20, 30)
                         }

        min_temp, max_temp = crop_temp_map[row['Crop Type']]

        return 1 if min_temp <= row['Temperature'] <= max_temp else 0

    scaler = MinMaxScaler()
    env_cols = ['Temperature', 'Humidity', 'Moisture']
    df[env_cols] = scaler.fit_transform(df[env_cols])

    df['env_max'] = df[env_cols].max(axis=1)
    df['temp_suitability'] = df.apply(is_temp_suitable, axis=1)

    return df


def load_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """ Prepare training and test data into pandas DataFrames: added columns, transformations, etc.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training, test and original data, ready for model
            training.
    """
    train = clean_data(pd.read_csv('train.csv'))
    original = clean_data(pd.read_csv('Fertilizer_Prediction.csv'))
    test = clean_data(pd.read_csv('test.csv'))

    train['is_original'] = pd.Categorical(['False'] * len(train), categories=['False', 'True'])
    original['is_original'] = pd.Categorical(['True'] * len(original), categories=['False', 'True'])
    test['is_original'] = pd.Categorical(['False'] * len(test), categories=['False', 'True'])

    # constant feature to add variety to tree ensemble
    train['constant'] = pd.Categorical(['const'] * len(train))
    original['constant'] = pd.Categorical(['const'] * len(original))
    test['constant'] = pd.Categorical(['const'] * len(test))

    #train = add_derived_cols(train)
    #test = add_derived_cols(test)

    train, test, original, encoder = encode_category_columns(train, test, original)

    #train = add_nutrient_chemistry(train)
    #test = add_nutrient_chemistry(test)

    return train, test, original, encoder

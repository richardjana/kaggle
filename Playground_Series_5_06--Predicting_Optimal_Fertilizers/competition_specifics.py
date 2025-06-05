from itertools import combinations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
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

    if framework == 'LGBM':
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
            try:
                test_df[col] = le.transform(test_df[col])
            except:
                pass
            label_encoders[col] = le
        
        return train_df, test_df, label_encoders

    #if framework == 'XGB':

    #if framework == 'TF':


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

    for col in ['Height', 'Weight', 'Heart_Rate']:
        pd_df[f"{col}_zscore"] = pd_df.groupby('Sex')[col].transform(zscore)

    return pd_df


def generate_extra_columns(pd_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """ Generate extra feature columns from the original data, by combining columns.
    Args:
        pd_df (pd.DataFrame): The original data.
    Returns:
        pd.DataFrame: DataFrame with new columns added.
    """
    combine_cols = [col for col in pd_df.keys() if col != target_col]

    new_cols = {}

    for n in [2, 3, 4]:
        for cols in combinations(combine_cols, n):
            col_name = '*'.join(cols)
            new_cols[col_name] = pd_df[list(cols)].prod(axis=1)

    for cols in combinations(combine_cols, 2):
        col_name = '/'.join(cols)
        new_cols[col_name] = pd_df[cols[0]] / pd_df[cols[1]]

    pd_df = pd.concat([pd_df, pd.DataFrame(new_cols, index=pd_df.index)], axis=1)

    return pd_df


def load_preprocess_data(train_csv: str, test_csv: str, target_col: str, framework: str,
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """ Prepare training and test data into pandas DataFrames: added columns, transformations, etc.
    Args:
        train_csv (str): Name of the training file.
        test_csv (str): Name of the test file.
        target_col (str): Name of the target column.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and test data, ready for model training.
    """
    train = clean_data(pd.read_csv(train_csv))
    test = clean_data(pd.read_csv(test_csv))

    train, test, encoders = encode_category_columns(train, test, framework)

    #train = generate_extra_columns(train, target_col)
    #test = generate_extra_columns(test, target_col)

    #train = add_intuitive_columns(train)
    #test = add_intuitive_columns(test)

    return train, test, encoders

from itertools import combinations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold
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

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(train_df, train_df[target_col]):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        encoding_table = compute_target_encoding(fold_train, encode_col, target_col)
        fold_val_encoded = apply_target_encoding(fold_val, encode_col, encoding_table)
        train_encoded.iloc[val_idx] = fold_val_encoded.values

    train_df[encoded_cols] = train_encoded.astype(float)
    train_df = train_df.drop(columns=[encode_col, encoded_cols[0]])

    return train_df, test_df


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
    #from sklearn.model_selection import train_test_split
    #train, _ = train_test_split(train, test_size=0.9)
    test = clean_data(pd.read_csv(test_csv))

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



"""
Nutrient chemistry
    Ratios: N/P, N/K, P/K, plus total N+P+K. 
"""

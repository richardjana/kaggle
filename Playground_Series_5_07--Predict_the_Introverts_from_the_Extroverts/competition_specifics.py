from itertools import combinations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.experimental import enable_iterative_imputer # Required for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


TARGET_COL = 'Personality'
COMPETITION_NAME = 'playground-series-s5e7'


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

    for cat_col in ['Stage_fear', 'Drained_after_socializing']:
        pd_df[cat_col] = pd_df[cat_col].fillna('unknown').astype('category')

    #for num_col in ['Time_spent_Alone',  'Social_event_attendance', 'Going_outside',
    #                'Friends_circle_size', 'Post_frequency']:
    #    # .astype('category')#
    #    pd_df[num_col] = pd_df[num_col].fillna(pd_df[num_col].median()).astype('int')

    return pd_df


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


def load_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """ Prepare training and test data into pandas DataFrames: added columns, transformations, etc.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training, test and original data, ready for model
            training.
    """
    train = clean_data(pd.read_csv('train.csv'))
    test = clean_data(pd.read_csv('test.csv'))

    # constant feature to add variety to tree ensemble
    #train['constant'] = pd.Categorical(['const'] * len(train))
    #test['constant'] = pd.Categorical(['const'] * len(test))

    label_encoder = LabelEncoder()
    train[TARGET_COL] = label_encoder.fit_transform(train[TARGET_COL])

    num_cols = ['Time_spent_Alone',  'Social_event_attendance', 'Going_outside',
                'Friends_circle_size', 'Post_frequency']
    imputer = IterativeImputer(max_iter=10, random_state=42)
    train[num_cols] = imputer.fit_transform(train[num_cols])
    test[num_cols] = imputer.transform(test[num_cols])

    return train, test, label_encoder

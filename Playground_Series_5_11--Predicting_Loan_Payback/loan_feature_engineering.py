from itertools import combinations
from typing import List

import pandas as pd


def create_frequency_features(df, df_test, NUMS, CATS):
    """
    Add frequency and binning features efficiently.

    - For each categorical column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5, 10, 15 quantile bins.
    """
    # Pre-allocate DataFrames for new features to avoid fragmentation
    freq_features_train = pd.DataFrame(index=df.index)
    freq_features_test = pd.DataFrame(index=df_test.index)
    bin_features_train = pd.DataFrame(index=df.index)
    bin_features_test = pd.DataFrame(index=df_test.index)

    for col in NUMS+CATS:
        # --- Frequency encoding ---
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq)
        freq_features_test[f"{col}_freq"] = df_test[col].map(freq).fillna(freq.mean())

        # --- Quantile binning for numeric columns ---
        if col in NUMS:
            for q in [5, 10, 15]:
                try:
                    train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
                    bin_features_train[f"{col}_bin{q}"] = train_bins
                    bin_features_test[f"{col}_bin{q}"] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    bin_features_train[f"{col}_bin{q}"] = 0
                    bin_features_test[f"{col}_bin{q}"] = 0

    # Concatenate all new features at once
    df = pd.concat([df, freq_features_train, bin_features_train], axis=1)
    df_test = pd.concat([df_test, freq_features_test, bin_features_test], axis=1)

    return df, df_test



"""Extract the numeric subgrade value from the alphanumeric string column."""
def add_subgrade_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['subgrade'] = df['grade_subgrade'].str[1:].astype(int)
    df['grade'] = df['grade_subgrade'].str[0].astype('category')

    return df


def add_biagram_feature(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col1, col2 in combinations(cols, 2):
        new_col_name = f'{col1}_{col2}'
        df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
        df[new_col_name] = df[new_col_name].astype('category')

    return df

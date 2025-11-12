from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def apply_groupby_stats(data: List[pd.DataFrame], cats: List[str],
                        nums: List[str], stats: Optional[List[str]]
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Similar to Target Encoding and Count Encoding, this groups by one column and computes
        statistics on another column. Would need a nested skf loop, if the target was in nums!
    Args:
        data (List[pd.DataFrame]): Data: train, validation and test.
        cats (List[str]): Categorical columns to group by.
        nums (List[str]): Numerical columns to calculate stats on.
        stats (Optional[List[str]]): Statistics to compute. Defaults to ['mean', 'std'].
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation and test features
            with added columns.
    """
    if stats is None:
        stats = ['mean', 'std']

    X_train, X_valid, X_test = [df.copy() for df in data]

    for cat in cats:
        for num in nums:
            for stat in stats:
                grouped = X_train.groupby(cat)[num].transform(stat)
                col_name = f"GROUP_{cat}_{num}_{stat}"
                X_train[col_name] = grouped

                mapping = X_train[[cat, col_name]].drop_duplicates().set_index(cat)[col_name]
                X_valid[col_name] = X_valid[cat].map(mapping)
                X_test[col_name] = X_test[cat].map(mapping)

    return X_train, X_valid, X_test


def add_groupby_histogram(data: List[pd.DataFrame], skf: StratifiedKFold,
                          col: str, target: str, bins: int = 10
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """_summary_
    Args:
        data (List[pd.DataFrame]): Data: train, validation and test.
        skf (StratifiedKFold): K-fold splitter to nest, to avoid information leakage.
        col (str): Column name to group by.
        target (str): Name of the target column.
        bins (int, optional): Number of histogram bins. Defaults to 10.
    """
    def make_histogram(targets, bins=bins, range_min=0, range_max=1):
        hist, _ = np.histogram(targets, bins=bins, range=(range_min, range_max))
        return hist

    X_train, X_valid, X_test = [df.copy() for df in data]

    hist_cols = [f"histogram_{x}" for x in range(bins)]
    X_train[hist_cols] = np.nan  # Create empty columns

    # create training features in nested loop
    for train_idx, valid_idx in skf.split(X_train, X_train[target]):
        X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]

        hist_map = X_train_fold.groupby(col)[target].apply(make_histogram)
        hist_df = pd.DataFrame(hist_map.tolist(), index=hist_map.index, columns=hist_cols)

        X_valid_fold = X_valid_fold.merge(hist_df, on=col, how='left')
        X_train.iloc[valid_idx, X_train.columns.get_indexer(hist_cols)] = X_valid_fold[hist_cols].values

    # create validation and test features on full X_train
    full_hist_map = X_train.groupby(col)[target].apply(make_histogram)
    full_hist_df = pd.DataFrame(full_hist_map.tolist(), index=full_hist_map.index, columns=hist_cols)

    X_valid = X_valid.merge(full_hist_df, on=col, how='left')
    X_test = X_test.merge(full_hist_df, on=col, how='left')

    return X_train, X_valid, X_test


def add_groupby_quantiles(data: List[pd.DataFrame], skf: StratifiedKFold, col: str,
                          target: str, quantiles: Optional[List[float]] = None
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Add target quantile features, based on groups by a column.
    Args:
        data (List[pd.DataFrame]): Data: train, validation and test.
        skf (StratifiedKFold): K-fold splitter to nest, to avoid information leakage.
        col (str): Column name to group by.
        target (str): Name of the target column.
        quantiles (Optional[List[float]], optional): List of quantiles to add. Defaults to None.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Data with added features.
    """
    if quantiles is None:
        quantiles = [0.05, 0.10, 0.40, 0.45, 0.55, 0.60, 0.90, 0.95]

    X_train, X_valid, X_test = [df.copy() for df in data]

    quant_func_dict = {f"quantile_{int(q*100)}": (lambda x, q=q: x.quantile(q)) for q in quantiles}
    quant_cols = list(quant_func_dict.keys())
    X_train[quant_cols] = np.nan  # Create empty columns

    # create training features in nested loop
    for train_idx, valid_idx in skf.split(X_train, X_train[target]):
        X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]

        quant_df = X_train_fold.groupby(col)[target].agg(quant_func_dict).reset_index()

        X_valid_fold = X_valid_fold.merge(quant_df, on=col, how='left')
        X_train.iloc[valid_idx, X_train.columns.get_indexer(quant_cols)] = X_valid_fold[quant_cols].values

    # create validation and test features on full X_train
    full_quant_df = X_train.groupby(col)[target].agg(quant_func_dict).reset_index()

    X_valid = X_valid.merge(full_quant_df, on=col, how='left')
    X_test = X_test.merge(full_quant_df, on=col, how='left')

    return X_train, X_valid, X_test

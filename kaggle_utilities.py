from typing import Dict, List

from collections.abc import Callable
import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import seaborn as sns
import sklearn

matplotlib.use('Agg')

def make_category_error_plot(pd_df: pd.DataFrame, target_col: str,
                             fname: str, n_categories: int) -> None:
    """ Make error plot for prediction of categories.
    Args:
        pd_df (pd.DataFrame): Pandas DataFrame with 'id' (dummy), target_col and 'PREDICTION'
            columns.
        target_col (str): Name of the target column.
        fname (str): File name for the produced plot.
        n_categories (int): Number of categories in the data.
    """
    h_map = np.zeros((n_categories, n_categories))
    counts = pd_df[['id', target_col, 'PREDICTION']].groupby(
        [target_col, 'PREDICTION'], as_index=False).count().values.tolist()
    for r, p, c in counts:
        h_map[p, r] = c

    h_map /= np.sum(h_map)

    cmap = sns.color_palette('rocket', as_cmap=True)
    chart = sns.heatmap(h_map, cmap=cmap, square=True, linewidths=.5, cbar_kws={'shrink': .5})

    for y in range(n_categories):
        for x in range(n_categories):
            txt = plt.text(x + 0.5, y + 0.5, f"{h_map[y, x]:.3f}", ha='center', va='center')
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w'),
                                  PathEffects.withStroke(linewidth=1, foreground='k')])

    chart.invert_yaxis()
    chart.set_xlabel(target_col)
    chart.set_ylabel(f"Predicted {target_col}")

    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def make_diagonal_plot(train: pd.DataFrame,
                       val: pd.DataFrame,
                       target_col: str,
                       metric: Callable[[List[float], List[float]], float],
                       metric_name: str,
                       fname: str) -> None:
    """ Make a diagonal error plot, showing both training and validation data points.
    Args:
        train (pd.DataFrame): Training data.
        val (pd.DataFrame): Validation data.
        target_col (str): Name of the target column.
        metric (Callable[[List[float], List[float]], float]): Function to calculate the evaluation
            metric.
        metric_name (str): Name of the evaluation metric.
        fname (str): File name for the plot image.
    """
    chart = sns.scatterplot(data=train, x=target_col, y='PREDICTION', alpha=0.25)
    sns.scatterplot(data=val, x=target_col, y='PREDICTION', alpha=0.25)

    min_val = min(chart.get_xlim()[0], chart.get_ylim()[0])
    max_val = max(chart.get_xlim()[1], chart.get_ylim()[1])
    chart.set_xlim([min_val, max_val])
    chart.set_ylim([min_val, max_val])
    chart.plot([min_val, max_val], [min_val, max_val], linewidth=1, color='k')

    chart.set_aspect('equal')
    chart.set_xlabel(target_col)
    chart.set_ylabel(f"Predicted {target_col}")

    metric_value = metric(train[target_col], train['PREDICTION'])
    labels = [f"training ({metric_value:.2f})"]
    metric_value = metric(val[target_col], val['PREDICTION'])
    labels += [f"validation ({metric_value:.2f})"]
    plt.legend(labels=labels, title=f"dataset ({metric_name}):", loc='best')

    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def make_training_plot(history: Dict[str, List[int]], fname: str) -> None:
    """ Make plot to visualize the training progress.
    Args:
        history (Dict[str, List[int]]): History from model.fit.
        fname (str): File name for the plot image.
    """
    metric = list(history.keys())[0]

    _, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
    ax.plot(np.arange(len(history[metric]))+1, history[metric], 'r', label=f"training {metric}")
    ax.plot(np.arange(len(history[f"val_{metric}"]))+1, history[f"val_{metric}"],
            'g', label=f"validation {metric}")
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    plt.legend(loc='best')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def make_ROC_plot(pd_df: pd.DataFrame, target_col: str, fname: str) -> None:
    """ Make Receiver Operating Characteristic (ROC) plot with Area Under the Curve (AUC) value.
    Args:
        pd_df (pd.DataFrame): Pandas DataFrame with 'PREDICTION_PROBABILITY' and target_col
            columns.
        target_col (str): Name of the target column.
        fname (str): File name for the plot image.
    """
    fpr, tpr, _ = sklearn.metrics.roc_curve(pd_df[target_col], pd_df['PREDICTION_PROBABILITY'])
    auc = sklearn.metrics.roc_auc_score(pd_df[target_col], pd_df['PREDICTION_PROBABILITY'])

    _, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
    ax.plot([0, 1], [0, 1], '--k')
    ax.plot(fpr, tpr)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    plt.text(0.5, 0.5, f"AUC = {auc:.5f}", ha='center', va='center', backgroundcolor='white')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def min_max_scaler(df_list: List[pd.DataFrame], col_names: List[str]) -> List[pd.DataFrame]:
    """ Scale a set of columns from a list of pandas DataFrames to the [0, 1] range.
    Args:
        df_list (List[pd.DataFrame]): List of DataFrames.
        col_names (List[str]): List of column names.
    Returns:
        List[pd.DataFrame]: List of the scaled DataFrames.
    """

    for col in col_names:
        min_val = np.inf
        max_val = -np.inf
        for df in df_list:
            min_val = min(min_val, df[col].min())
            max_val = max(max_val, df[col].max())
        for df in df_list:
            df[col] = (df[col]-min_val) / (max_val-min_val)

    return df_list

def RMSE(arr_1: NDArray, arr_2: NDArray) -> float:
    """ Calculate the Root Mean Squared Error (RMSE) between two arrays.
    Args:
        arr_1 (NDArray): First array.
        arr_2 (NDArray): Second array.
    Returns:
        float: The RMSE.
    """
    return round(np.sqrt(np.sum(np.power(arr_1-arr_2, 2))/arr_1.size), 3)

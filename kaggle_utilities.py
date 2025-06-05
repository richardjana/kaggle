from typing import Dict, List, Tuple

from collections.abc import Callable
import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import xgboost as xgb

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
    chart = sns.heatmap(h_map, cmap=cmap, square=True,
                        linewidths=.5, cbar_kws={'shrink': .5})

    for y in range(n_categories):
        for x in range(n_categories):
            txt = plt.text(x + 0.5, y + 0.5,
                           f"{h_map[y, x]:.3f}", ha='center', va='center')
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
                       fname: str,
                       precision: int = 2) -> None:
    """ Make a diagonal error plot, showing both training and validation data points.
    Args:
        train (pd.DataFrame): Training data.
        val (pd.DataFrame): Validation data.
        target_col (str): Name of the target column.
        metric (Callable[[List[float], List[float]], float]): Function to calculate the evaluation
            metric.
        metric_name (str): Name of the evaluation metric.
        fname (str): File name for the plot image.
        precision (int): Number of decimals to print for the metric. Defaults to 2.
    """
    chart = sns.scatterplot(data=train, x=target_col,
                            y='PREDICTION', alpha=0.25)
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
    labels = [f"training ({metric_value:.{precision}f})"]
    metric_value = metric(val[target_col], val['PREDICTION'])
    labels += [f"validation ({metric_value:.{precision}f})"]
    plt.legend(labels=labels, title=f"dataset ({metric_name}):", loc='best')

    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def make_training_plot(history: Dict[str, List[float]], fname: str, precision: int = 2) -> None:
    """ Make plots to visualize the training progress: y-axis 1) linear scale 2) log scale.
    Args:
        history (Dict[str, List[float]]): History from model.fit.
        fname (str): File name for the plot image.
        precision (int): Number of decimals to print for the metric. Defaults to 2.
    """
    metric = list(history.keys())[0]

    _, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
    ax.plot(np.arange(len(history[metric]))+1, history[metric], 'r',
            label=f"training {metric} ({min(history[metric]):.{precision}f})")
    ax.plot(np.arange(len(history[f"val_{metric}"]))+1, history[f"val_{metric}"], 'g',
            label=f"validation {metric} ({min(history[f'val_{metric}']):.{precision}f})")
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    plt.legend(loc='best')
    plt.savefig(fname, bbox_inches='tight')

    ax.set_yscale('log')
    plt.savefig(f"{fname[:-4]}_LOG.png", bbox_inches='tight')

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


def RMSE(arr_1: np.ndarray, arr_2: np.ndarray) -> float:
    """ Calculate the Root Mean Squared Error (RMSE) between two arrays.
    Args:
        arr_1 (NDArray): First array.
        arr_2 (NDArray): Second array.
    Returns:
        float: The RMSE.
    """
    return round(np.sqrt(np.sum(np.power(arr_1-arr_2, 2))/arr_1.size), 3)


def rmsle_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ Compute the Root Mean Squared Logarithmic Error (RMSLE) between true and predicted values.
    Args:
        y_true (Tensor): Ground truth values (non-negative).
        y_pred (Tensor): Predicted values (non-negative).
    Returns:
        Tensor: A scalar tensor containing the RMSLE.
    """
    # Clip values to avoid log(0); assume values must be >= 0
    y_true = tf.clip_by_value(y_true, 0.0, tf.float32.max)
    y_pred = tf.clip_by_value(y_pred, 0.0, tf.float32.max)

    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Compute the Root Mean Squared Logarithmic Error (RMSLE) between true and predicted values.
    Args:
        y_true (np.ndarray): Ground truth values (non-negative).
        y_pred (np.ndarray): Predicted values (non-negative).
    Returns:
        float: RMSLE score.
    """
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    return np.sqrt(np.mean((log_pred - log_true) ** 2))


def mapk(actual: List[List[int]], predicted: List[List[int]], k: int = 3) -> float:
    """ Mean Average Precision @ K (MAP@K) between actual and predicted labels.
    Args:
        actual (list[list[int]]): Ground truth labels.
        predicted (list[list[int]]): Predicted labels.
        k (int, optional): Number of evaluated predictions. Defaults to 3.
    Returns:
        float: MAP@K value.
    """
    def apk(a, p, k: int) -> float:
        """ Average Precision @ K - score for a single prediction.
        Args:
            a (List[int]): Actual label(s).
            p (List[int]): Predicted labels(s).
            k (int): Number of predicted labels.
        Returns:
            float: Score.
        """
        if len(p) > k:
            p = p[:k]

        score = 0.0
        num_hits = 0.0
        for i, pred in enumerate(p):
            if pred in a and pred not in p[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score / min(len(a), k) if a else 0.0

    return float(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]))


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, file_name: str,
                          class_names: List[str], normalize: bool = True) -> None:
    """
    Plot a confusion matrix using matplotlib.
    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        file_name (str): File name for the saved plot.
        class_names (list[str], optional): List of class names (labels). Defaults to None.
        normalize (bool): Whether to normalize by row (true label counts). Defaults to True.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    _, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format=".2f" if normalize else "d", xticks_rotation=45)

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

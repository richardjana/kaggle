import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def make_category_error_plot(pd_df, target_col, fname, n_categories):
    h_map = np.zeros((n_categories, n_categories))
    counts = pd_df[['id', target_col, 'PREDICTION']].groupby([target_col, 'PREDICTION'], as_index=False).count().values.tolist()
    for r, p, c in counts:
        h_map[p, r] = c

    h_map /= np.sum(h_map)

    cmap = sns.color_palette('rocket', as_cmap=True)
    chart = sns.heatmap(h_map, cmap=cmap, square=True, linewidths=.5, cbar_kws={'shrink': .5})

    for y in range(n_categories):
        for x in range(n_categories):
            txt = plt.text(x + 0.5, y + 0.5, f"{h_map[y, x]:.3f}", ha='center', va='center')
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w'), PathEffects.withStroke(linewidth=1, foreground='k')])

    chart.invert_yaxis()
    chart.set_xlabel(target_col)
    chart.set_ylabel(f"Predicted {target_col}")

    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def make_diagonal_plot(train, val, target_col, metric, metric_name, fname):
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

    RMSE = metric(train[target_col], train['PREDICTION'])
    labels = [f"training ({RMSE:.2f})"]
    RMSE = metric(val[target_col], val['PREDICTION'])
    labels += [f"validation ({RMSE:.2f})"]
    plt.legend(labels=labels, title=f"dataset ({metric_name}):", loc='best')

    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def make_training_plot(history, fname):
    metric = list(history.keys())[0]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
    ax.plot(np.arange(len(history[metric]))+1, history[metric], 'r', label=f"training {metric}")
    ax.plot(np.arange(len(history[f"val_{metric}"]))+1, history[f"val_{metric}"], 'g', label=f"validation {metric}")
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    plt.legend(loc='best')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

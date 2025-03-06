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
    plt.close()

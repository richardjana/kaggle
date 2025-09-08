from glob import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

PATTERN = sys.argv[1]
TOP_PCT = [0.50, 0.25, 0.10]
COLORS = ['red', 'green', 'blue']
ALPHAS = [0.50, 0.75, 1.0]
N_BINS = 50

### load studies and extract completed trials into combined DataFrame
all_dfs = []
for study_file in glob(PATTERN):
    study = optuna.load_study(storage=f"sqlite:///{study_file}",
                              study_name='bearsbeatbeets')

    study_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    #study_df['source'] = study_file.split('/')[0].split('--')[1]  # add source

    all_dfs.append(study_df[study_df['state'] == 'COMPLETE'])  # filter completed trials

df = pd.concat(all_dfs, ignore_index=True)

print(f"Plotting {len(df)} trials from {len(all_dfs)} studies.")

### clean and sort DataFrame
df.sort_values(by='value', inplace=True)
df.drop('state', axis=1, inplace=True)  # no longer needed

PREFIX = 'params_'  # rename params columns (remove 'params_' prefix)
params_cols_map = {col: col.replace(PREFIX, '') for col in df.columns if col.startswith(PREFIX)}
df.rename(columns=params_cols_map, inplace=True)

# filter top-performing trials
top_dfs = {tp: df.head(int(len(df)*tp)) for tp in TOP_PCT}

### iterate through parameters and plot distributions
for param in params_cols_map.values():
    if pd.api.types.is_numeric_dtype(df[param]):
        unique_vals = df[param].unique()
        is_discrete = len(unique_vals) <= N_BINS

        plt.figure(figsize=(8, 4))

        if is_discrete:
            sorted_vals = sorted(unique_vals)

            total_counts = df[param].value_counts().sort_index()
            for tp, color, alpha in zip(TOP_PCT, COLORS, ALPHAS):
                top_counts = top_dfs[tp][param].value_counts().sort_index()
                frac_good = (top_counts / total_counts).reindex(sorted_vals, fill_value=0)

                plt.plot(sorted_vals, frac_good.values, 'o:', label=f"Top {tp*100:.0f}%", 
                        color=color, alpha=alpha)

            plt.xlabel(param)
            plt.xticks(sorted_vals)

        else:
            bins = np.linspace(df[param].min(), df[param].max(), N_BINS + 1)
            df['bin'] = pd.cut(df[param], bins=bins, include_lowest=True)

            for tp, color, alpha in zip(TOP_PCT, COLORS, ALPHAS):
                top_df = top_dfs[tp].copy()
                top_df['bin'] = pd.cut(top_df[param], bins=bins, include_lowest=True)

                bin_counts = df['bin'].value_counts().sort_index()
                bin_top_counts = top_df['bin'].value_counts().sort_index()
                frac_good = (bin_top_counts / bin_counts).fillna(0)

                bin_centers = [interval.mid for interval in frac_good.index]
                plt.plot(bin_centers, frac_good.values, 'o:', label=f"Top {tp*100:.0f}%", 
                        color=color, alpha=alpha)

            plt.xlabel(param)
            df.drop(columns='bin', inplace=True)


        x_min, x_max = plt.xlim()
        plt.xlim(x_min, x_max)

        for tp, color, alpha in zip(TOP_PCT, COLORS, ALPHAS):
            plt.plot([x_min, x_max], [tp, tp], '-', color=color)

        plt.xlabel(param)
        plt.ylabel('Fraction of Good Trials')
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"{param}_distribution.png")
        plt.close()

    else:
        print(f"Skipping non-numeric parameter: {param}")

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
        N_BINS = 50
        plt.figure(figsize=(8, 4))

        # Bin all data
        bins = np.linspace(df[param].min(), df[param].max(), N_BINS + 1)
        df['bin'] = pd.cut(df[param], bins=bins, include_lowest=True)

        for tp, color, alpha in zip(TOP_PCT, COLORS, ALPHAS):
            top_df = top_dfs[tp].copy()
            top_df['bin'] = pd.cut(top_df[param], bins=bins, include_lowest=True)

            # Group and compute fraction of good trials in each bin
            bin_counts = df['bin'].value_counts().sort_index()
            bin_top_counts = top_df['bin'].value_counts().sort_index()
            fraction_good = (bin_top_counts / bin_counts).fillna(0)

            bin_centers = [interval.mid for interval in fraction_good.index]

            plt.plot(bin_centers, fraction_good.values, 'o:', label=f"Top {tp*100:.0f}%",
                     color=color, alpha=alpha)

        x_min, x_max = plt.xlim()
        plt.xlim(x_min, x_max)

        for tp, color, alpha in zip(TOP_PCT, COLORS, ALPHAS):
            plt.plot([x_min, x_max], [tp, tp], '-', color=color)

        plt.title(f"Fraction of Good Trials vs. '{param}'")
        plt.xlabel(param)
        plt.ylabel('Fraction of Good Trials')
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"{param}_distribution.png")
        plt.close()

    else:
        print(f"Skipping non-numeric parameter: {param}")

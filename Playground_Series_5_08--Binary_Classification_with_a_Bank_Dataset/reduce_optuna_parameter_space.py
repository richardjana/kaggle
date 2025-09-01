import sys

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns

ALL_DBS = sys.argv[1:]
TOP_PCT = [0.50, 0.25]
COLORS = ['red', 'green']
ALPHAS = [0.75, 1.0]

### load studies and extract completed trials into combined DataFrame
all_dfs = []
for study_file in ALL_DBS:
    study = optuna.load_study(storage=f"sqlite:///{study_file}",
                              study_name='banking')

    study_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    #study_df['source'] = study_file.split('/')[0].split('--')[1]  # add source

    all_dfs.append(study_df[study_df['state'] == 'COMPLETE'])  # filter completed trials

df = pd.concat(all_dfs, ignore_index=True)

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
        plt.figure(figsize=(8, 4))
        sns.histplot(df[param], kde=True, bins=20, color='lightgray', alpha=0.5, label='All trials')
        for tp, color, alpha in zip(TOP_PCT, COLORS, ALPHAS):
            sns.histplot(top_dfs[tp][param], kde=True, bins=20, color=color, alpha=alpha,
                        label=f"Top {tp*100:.0f}%")
        plt.title(f"Distribution of '{param}'")
        plt.xlabel(param)
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"{param}_distribution.png")
        plt.close()
    else:
        print(f"Skipping non-numeric parameter: {param}")

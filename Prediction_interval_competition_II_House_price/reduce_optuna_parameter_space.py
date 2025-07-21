import os
import sys

from glob import glob
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns

### load studies and extract completed trials into combined DataFrame
all_dfs = []
#for study_file in glob('*/optuna_study_mean.db'):
for study_file in glob('*/optuna_study_residual.db'):
    study = optuna.load_study(storage=f"sqlite:///{study_file}",
                              study_name='house_price_residual')
                              #study_name='house_price_mean')

    study_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    study_df['source'] = study_file.split('/')[0].split('--')[1]  # add source

    all_dfs.append(study_df[study_df['state'] == 'COMPLETE'])  # filter completed trials

df = pd.concat(all_dfs, ignore_index=True)

### clean and sort DataFrame
df.sort_values(by='value', inplace=True)
df.drop('state', axis=1, inplace=True)  # no longer needed

PREFIX = 'params_'  # rename params columns (remove 'params_' prefix)
params_cols_map = {col: col.replace(PREFIX, '') for col in df.columns if col.startswith(PREFIX)}
df.rename(columns=params_cols_map, inplace=True)

TOP_PCT = 0.5  # filter top-performing trials
n_top = int(len(df) * TOP_PCT)
top_df = df.head(n_top)

# --- Step 4: Iterate through parameters and plot distributions ---

for param in params_cols_map.values():
    if pd.api.types.is_numeric_dtype(top_df[param]):
        plt.figure(figsize=(8, 4))
        sns.histplot(df[param], kde=True, bins=20, color='lightgray', label='All trials', alpha=0.5)
        sns.histplot(top_df[param], kde=True, bins=20, color='blue', label=f"Top {TOP_PCT*100:.0f}%")
        plt.title(f"Distribution of '{param}'\n(All Trials vs. Top {TOP_PCT*100:.0f}%)")
        plt.xlabel(param)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"{param}_distribution.png")
        plt.close()
    else:
        print(f"Skipping non-numeric parameter: {param}")

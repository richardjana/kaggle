import os
import sys

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns

# --- Step 1: Load study and extract completed trials ---
study = optuna.load_study(study_name='house_price_mean', storage='sqlite:///optuna_study.db')
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

# --- Step 2: Convert trials to DataFrame ---
trial_dicts = [{**t.params, 'value': t.value} for t in completed_trials]
df = pd.DataFrame(trial_dicts).sort_values(by='value')

# --- Step 3: Filter top-performing trials (e.g. top 50%) ---
top_pct = 0.5  # 50%
n_top = int(len(df) * top_pct)
top_df = df.head(n_top)

# --- Step 4: Iterate through parameters and plot distributions ---
param_columns = [col for col in top_df.columns if col != 'value']

for param in param_columns:
    if pd.api.types.is_numeric_dtype(top_df[param]):
        plt.figure(figsize=(8, 4))
        sns.histplot(df[param], kde=True, bins=20, c='lightgray', label='All trials', alpha=0.5)
        sns.histplot(top_df[param], kde=True, bins=20, c='blue', label=f"Top {top_pct*100:.0f}%")
        plt.title(f"Distribution of '{param}'\n(All Trials vs. Top {top_pct*100:.0f}%)")
        plt.xlabel(param)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"{param}_distribution.png")
        plt.close()
    else:
        print(f"Skipping non-numeric parameter: {param}")


# provide copy-paste-able format
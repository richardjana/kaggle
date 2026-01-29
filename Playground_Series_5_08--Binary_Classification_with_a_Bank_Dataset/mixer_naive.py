import pandas as pd

submissions = []
for model in ['XGB', 'LGBM']:
	submissions.append(pd.read_csv(f"Playground_Series_5_08--Binary_Classification_with_a_Bank_Dataset_12/predictions_{model}_optuna.csv"))

mixed = submissions[0]
for sub in submissions[1:]:
	mixed['y'] += sub['y']
mixed['y'] /= len(submissions)

mixed.to_csv('predictions_mixed.csv', columns=['id', 'y'], index=False)

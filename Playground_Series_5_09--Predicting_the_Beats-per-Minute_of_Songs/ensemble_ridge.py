from glob import glob
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error


def evaluate_single_models(dirs: List[str], oofs: List[pd.Series], y_true: pd.Series) -> None:
    """ For each of the individual models used in the ensemble, print the OOF RMSE,
        in descending order.
    Args:
        dirs (List[str]): Directories / names to identify the models.
        oofs (List[pd.Series]): OOF predictions of all models.
        y_true (pd.Series): True values for the predictions.
    """
    rmses = np.array([root_mean_squared_error(y_true, oof) for oof in oofs])
    for index in np.argsort(-rmses):  # descending order
        print(f"{dirs[index]} RMSE: {rmses[index]:.5f}")

# patterns and file names
DIR_PATTERN = 'Playground_Series_5_09--Predicting_the_Beats-per-Minute_of_Songs_*'
OOF = 'oof.csv'
TEST = 'predictions_optuna.csv'

# gather data
dirs = glob(DIR_PATTERN)
oofs = [pd.read_csv(f"{d}/{OOF}")['oof'] for d in dirs]
y_trues = [pd.read_csv(f"{d}/{OOF}")['y_true'] for d in dirs]
test_preds = [pd.read_csv(f"{d}/{TEST}")['BeatsPerMinute'] for d in dirs]
ids = pd.read_csv(f"{dirs[0]}/{TEST}")['id']

# assert all oof are in consistent order
assert all(y_trues[0].equals(yt) for yt in y_trues[1:]), 'Not all OOFs are in the same order!'

# ridge regression ensemble
meta_model = Ridge()
meta_model.fit(np.column_stack(oofs), y_trues[0])

# evaluate the ensemble
evaluate_single_models(dirs, oofs, y_trues[0])
rmse = root_mean_squared_error(y_trues[0], meta_model.predict(np.column_stack(oofs)))
print(f"Ensemble Ridge OOF RMSE: {rmse:.5f}")

# make final predictions and write to file
final_preds = meta_model.predict(np.column_stack(test_preds))
pd.DataFrame({'id': ids, 'BeatsPerMinute': final_preds}).to_csv('ensemble_ridge.csv', index=False)

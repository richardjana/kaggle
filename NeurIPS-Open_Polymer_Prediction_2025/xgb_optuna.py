import sys
from typing import List, Tuple

import numpy as np
import optuna
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import xgboost as xgb


TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
N_CV_SPLITS = 5
try:
    I = int(sys.argv[1])  # index for the study (output files)
except IndexError:
    I = 0

### RDKit FE ###
def compute_all_descriptors(smiles: str) -> List[float | int | None]:
    """ Computes chemical descriptors from SMILES notation using RDKit.
    Args:
        smiles (str): SMILES descriptor for a structure.
    Returns:
        List[float | int | None]: List of descriptor values.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(Descriptors.descList)

    return [desc[1](mol) for desc in Descriptors.descList]


### read or create training data ###
def read_training_data(target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """ Try to read the training data with precomputed RDKit descriptors from file, otherwise
        read the raw training data and compute the descriptors. (And write them to file.)
    Args:
        target_col (str): Name of the target, to filter the sparse training data rows.
    Returns:
        Tuple[pd.DataFrame, List[str]]: The training data, with RDKit descriptors, and a list of
            the descriptor names.
    """
    try:
        train = pd.read_csv('train_rdkit_features.csv')
        desc_names = sorted(set(train.columns) - set(TARGETS))

    except (FileNotFoundError, pd.errors.EmptyDataError):
        train = pd.read_csv('train.csv')

        desc_names = [desc[0] for desc in Descriptors.descList]
        descriptors = [compute_all_descriptors(smi) for smi in train['SMILES'].to_list()]
        descriptors = pd.DataFrame(descriptors, columns=desc_names)
        train = pd.concat([train, descriptors],axis=1)

        train = train.drop(columns=['SMILES'])  # drop SMILES columns

        # drop RDKit descriptors that are missing for some reason or have dubious values
        cols = ['Ipc', 'BCUT2D_MWHI', 'BCUT2D_MWLOW',
                'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI',
                'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW']
        train = train.drop(columns=cols)
        desc_names = sorted(set(train.columns) - set(TARGETS))

        for col in desc_names:  # fix inf and NaN values
            train[col] = train[col].replace([np.inf, -np.inf], np.nan)
            median = train[col].median(skipna=True)
            train[col] = train[col].fillna(median)

        train.to_csv('train_rdkit_features.csv', index=False)

    train = train[train[target_col].notnull()]  # filter rows

    return train, desc_names


def objective(trial):
    """ Objective function for Optuna. """
    params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'n_estimators': 10_000,
        'random_state': 77,
        'verbosity': 0,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'max_delta_step': trial.suggest_float('max_delta_step', 1e-3, 10, log=True),
        'gamma': trial.suggest_float('gamma', 1e-3, 10, log=True),
        'use_label_encoder': False,
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 25, 25),
        'enable_categorical': False
    }

    kf = KFold(n_splits=N_CV_SPLITS, random_state=42, shuffle=True)
    oof_preds = np.zeros(train.shape[0])

    for train_idx, val_idx in kf.split(train):
        X_train_fold = train[desc_names].iloc[train_idx].values
        y_train_fold = train[TARGET_COL].iloc[train_idx].values
        X_val_fold = train[desc_names].iloc[val_idx].values
        y_val_fold = train[TARGET_COL].iloc[val_idx].values

        model = xgb.XGBRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  verbose=False
                  )

        oof_preds[val_idx] = model.predict(X_val_fold)

    return mean_absolute_error(train[TARGET_COL].values, oof_preds)


for TARGET_COL in TARGETS:
    train, desc_names = read_training_data(TARGET_COL)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10_000, timeout=60*60*0.5)

    worst_mae = max(trial.value for trial in study.trials if trial.value is not None)

    with open(f"optuna_XGB_{TARGET_COL}_{I}.txt", 'w', encoding='utf-8') as out_file:
        out_file.write(f"Best trial: {study.best_trial.number}/{len(study.trials)}\n")
        out_file.write(f"  MAE: {study.best_value}\n")
        out_file.write(f"  worst MAE: {worst_mae}\n")
        out_file.write('  Best hyperparameters:\n')
        for key, value in study.best_params.items():
            out_file.write(f"    {key}: {value}\n")

import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


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


ADDITIONAL_PARAMS = {'random_state': 77,
                     'n_jobs': -1,
                     'criterion': 'squared_error',
                     'bootstrap': True
                     }


def get_param_space(trial: optuna.trial.Trial, target_col: str) -> Dict[str, int | float]:
    if target_col == 'Tg':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'max_depth': trial.suggest_int('max_depth', 10, 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.2, 1.0),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 100, 1000),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5)
        }
    if target_col == 'FFV':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'max_depth': trial.suggest_int('max_depth', 10, 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.2, 1.0),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 100, 1000),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5)
        }
    if target_col == 'Tc':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'max_depth': trial.suggest_int('max_depth', 10, 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.2, 1.0),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 100, 1000),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5)
        }
    if target_col == 'Density':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'max_depth': trial.suggest_int('max_depth', 10, 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.2, 1.0),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 100, 1000),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5)
        }
    if target_col == 'Rg':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'max_depth': trial.suggest_int('max_depth', 10, 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.2, 1.0),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 100, 1000),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5)
        }
    return {}


def make_objective(train: pd.DataFrame, desc_names: List[str], target_col: str) -> Callable[[optuna.trial.Trial], float]:
    def objective(trial: optuna.trial.Trial) -> float:
        """ Objective function for Optuna. """
        params = get_param_space(trial, target_col)
        params.update(ADDITIONAL_PARAMS)

        kf = KFold(n_splits=N_CV_SPLITS, random_state=42, shuffle=True)
        oof_preds = np.zeros(train.shape[0])

        for train_idx, val_idx in kf.split(train):
            X_train_fold = train[desc_names].iloc[train_idx].values
            y_train_fold = train[target_col].iloc[train_idx].values
            X_val_fold = train[desc_names].iloc[val_idx].values

            model = ExtraTreesRegressor(**params)
            model.fit(X_train_fold, y_train_fold)
            oof_preds[val_idx] = model.predict(X_val_fold)

        return mean_absolute_error(train[target_col].values, oof_preds)

    return objective


for TARGET_COL in TARGETS:
    train, desc_names = read_training_data(TARGET_COL)

    study = optuna.create_study(direction='minimize',
                                study_name=TARGET_COL,
                                storage=f"sqlite:///optuna_study_ET--{TARGET_COL}.db")
    study.optimize(make_objective(train, desc_names, TARGET_COL),
                   n_trials=10_000,
                   timeout=60*60*1)

    worst_mae = max(trial.value for trial in study.trials if trial.value is not None)

    with open(f"optuna_ET_{TARGET_COL}_{I}.txt", 'w', encoding='utf-8') as out_file:
        out_file.write(f"Best trial: {study.best_trial.number}/{len(study.trials)}\n")
        out_file.write(f"  squared errer: {study.best_value}\n")
        out_file.write(f"  worst squared error: {worst_mae}\n")
        out_file.write('  Best hyperparameters:\n')
        for key, value in study.best_params.items():
            out_file.write(f"    {key}: {value}\n")

import glob
from pprint import pformat
from typing import Dict, Tuple

import numpy as np

TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

OTHER_PARAMS = {'tree_method': 'hist',
                'objective': 'reg:absoluteerror',
                'eval_metric': 'mae',
                'n_estimators': 10_000,
                'random_state': 77,
                'use_label_encoder': False,
                'enable_categorical': False}


def convert_numerical(s: str) -> int | float | str:
    """ Convert a string to integer or float safely, if possible.
    Args:
        s (str): Some string, might be a number.
    Returns:
        int | float | str: The string, converted if possible.
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def read_trial_results_file(file_name: str) -> Tuple[float, Dict[str, float | int]]:
    """ Read a results file from a single optuna study.
    Args:
        file_name (str): Name of the file.
    Returns:
        Tuple[float, Dict[str, float | int]]: Best score of the study and the params to achieve it.
    """
    score = 1e6
    params = {}
    with open(file_name, 'r', encoding='utf-8') as res_file:
        for line in res_file.readlines():
            key = line.split(':')[0].strip()
            val = convert_numerical(line.split(':')[1].strip())

            match key:
                case 'MAE':
                    score = float(val)
                case 'Best trial' | 'worst MAE' | 'Best hyperparameters':
                    pass
                case _:
                    params[key] = val

    return score, params


def compile_params_for_framework(fw_name: str) -> Dict[str, float]:
    """ Read all results files for a framework, compile the best params for each target and write
        them to file in a copyable format.
    Args:
        fw_name (str): Name of the framework, used to find the files.
    Returns:
        Dict[str, float]: Individual MAE for each target.
    """
    data = {target: {} for target in TARGETS}
    best_scores = {target: 1e6 for target in TARGETS}
    best_indices = {target: -1 for target in TARGETS}

    # read / filter data from file
    for results_file in glob.glob(f"optuna_{fw_name}_*_*.txt"):
        target_col = results_file.split('_')[2]
        i = int(results_file.split('_')[3].split('.')[0])
        score, params = read_trial_results_file(results_file)

        if score < best_scores[target_col]:
            best_scores[target_col] = score
            data[target_col] = params
            best_indices[target_col] = i

    # add additional setting entries
    for param_dict in data.values():
        param_dict.update(OTHER_PARAMS)

    with open(f"best_params_{fw_name}.txt", 'w', encoding='utf-8') as best_params_file:
        best_params_file.write("params = ")
        best_params_file.write(pformat(data))

    print('Best study indices:')
    print(best_indices)

    return best_scores


def calculate_weighted_mae(mae: Dict[str, float]) -> float:
    """
    Calculate weighted MAE (wMAE) score according to the competition formula
    Args:
        Dict[str, float]: individual MAEs for each target
    Returns:
        float: wMAE score
    """
    sample_counts = {'Tg': 511, 'FFV': 7030, 'Tc': 737, 'Density': 613, 'Rg': 614}  # train.csv
    value_ranges = {'Tg': 620.2797376,  # train.csv
                    'FFV': 0.55010467,
                    'Tc': 0.47750000000000004,
                    'Density': 1.092307675,
                    'Rg': 24.944550504999995}

    # Calculate weights
    sqrt_inv_n = {t: np.sqrt(1.0 / sample_counts[t]) for t in TARGETS}
    sum_sqrt_inv_n = sum(sqrt_inv_n.values())
    weights = {}
    for target in TARGETS:
        weights[target] = len(TARGETS) * sqrt_inv_n[target] / value_ranges[target] / sum_sqrt_inv_n

    return sum(weights[t] * mae[t] for t in TARGETS)


for framework in ['XGB']:
    individual_maes = compile_params_for_framework(framework)
    print(f"individual MAEs = {individual_maes}")
    print(f"expected wMAE = {calculate_weighted_mae(individual_maes):.4f}")

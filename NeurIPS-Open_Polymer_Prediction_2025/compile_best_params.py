from typing import Dict, Tuple

import glob
from pprint import pformat

TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

OTHER_PARAMS = {'tree_method': 'hist',
                'objective': 'reg:absoluteerror',
                'eval_metric': 'mae'}


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


def compile_params_for_framework(framework: str) -> None:
    """ Read all results files for a framework, compile the best params for each target and write
        them to file in a copyable format.
    Args:
        framework (str): Name of the framework, used to find the files.
    """
    data = {target: {} for target in TARGETS}
    best_scores = {target: 1e6 for target in TARGETS}

    # read / filter data from file
    for results_file in glob.glob(f"optuna_{framework}_*_*.txt"):
        target_col = results_file.split('_')[2]
        score, params = read_trial_results_file(results_file)

        if score < best_scores[target_col]:
            data[target_col] = params

    # add additional setting entries
    for param_dict in data.values():
        param_dict.update(OTHER_PARAMS)

    with open(f"best_params_{framework}.json", 'w', encoding='utf-8') as best_params_file:
        best_params_file.write("params = ")
        best_params_file.write(pformat(data))


for framework in ['XGB']:
    compile_params_for_framework(framework)

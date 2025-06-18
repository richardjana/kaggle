from typing import Tuple
import sys

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import mapk  # noqa
from kaggle_api_functions import submit_prediction
from competition_specifics import TARGET_COL, COMPETITION_NAME, load_preprocess_data


def lgb_map3_eval(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    """ Custom LightGBM eval function to compute MAP@3.
    Parameters:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Array of predictions.
    Returns:
        Tuple[str, float, bool]: metric name, value, higher_is_better
    """
    top_3 = np.argsort(-y_pred, axis=1)[:, :3]  # Compute top 3 predictions
    actual = [[label] for label in y_true]  # Format true labels for mapk

    return 'map@3', mapk(actual, top_3.tolist(), k=3), True


def make_prediction(model: lgb.LGBMClassifier, test_df: pd.DataFrame) -> None:
    """ Make a prediction for the test data, with a given model.
    Args:
        model (lgb.LGBMRegressor): Model used for the prediction.
        test_df (pd.DataFrame): DataFrame with the test data, pre-processed.
    """
    submit_df = pd.read_csv('sample_submission.csv')
    pred_proba = np.asarray(model.predict_proba(test_df.to_numpy()))
    top_3_indices = np.argsort(-pred_proba, axis=1)[:, :3]

    top_3_labels = np.array([
        encoder.inverse_transform(sample_top3)
        for sample_top3 in top_3_indices
    ])

    joined_top3 = [' '.join(labels) for labels in top_3_labels]

    submit_df[TARGET_COL] = joined_top3
    submit_df.to_csv('predictions_LGBM_optuna.csv',
                     columns=['id', TARGET_COL], index=False)


# Load dataset
train, test, encoder = load_preprocess_data()

best_params = {
        'objective': 'multiclass',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 7100,
        'learning_rate': 0.010716278263965403,
        'num_leaves': 133,
        'max_depth': 13,
        'min_child_samples': 45,
        'subsample': 0.5325650054723919,
        'colsample_bytree': 0.5161798255547566,
        'reg_alpha': 0.7695187791606709,
        'reg_lambda': 0.48879889452033864
    }

y = train.pop(TARGET_COL)

model = joblib.load('lgb_model.pkl')

# make prediction for the test data
make_prediction(model, test)

public_score = submit_prediction(COMPETITION_NAME, 'predictions_LGBM_optuna.csv',
                                 f"LGBM optuna {SERIAL_NUMBER} ()")
print(f'Public score: {public_score}')

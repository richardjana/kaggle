import sys

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import mapk  # noqa
from kaggle_api_functions import submit_prediction
from competition_specifics import TARGET_COL, COMPETITION_NAME, load_preprocess_data


def xgb_feval_map3(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Custom evaluation function for XGBClassifier to compute MAP@3.
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Array of predictions.
    Returns:
        float: Metric value.
    """
    top_3 = np.argsort(-y_pred, axis=1)[:, :3]
    actual = [[int(label)] for label in y_true]

    return mapk(actual, top_3.tolist(), k=3)


def make_prediction(model: xgb.XGBClassifier, test_df: pd.DataFrame) -> None:
    """ Make a prediction for the test data, with a given model.
    Args:
        model (xgb.XGBClassifier): Model used for the prediction.
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
    submit_df.to_csv('predictions_XGB_optuna.csv',
                     columns=['id', TARGET_COL], index=False)


# Load dataset
train, test, encoder = load_preprocess_data()
NUM_CLASSES = train[TARGET_COL].unique()

best_params = {
    "objective": "multi:softprob",
    "eval_metric": 'mlogloss',
    "num_class": NUM_CLASSES,
    "tree_method": "hist",
    "eta": 0.0054713056635367525,
    "max_depth": 15,
    "min_child_weight": 70,
    "subsample": 0.819617292814055,
    "colsample_bytree": 0.5405932267716387,
    "reg_alpha": 0.018704027081724698,
    "reg_lambda": 0.7562848707375666,
    "n_estimators": int(3318 * np.sqrt(1/0.5)),
    "use_label_encoder": False,
}

# Train final model with best parameters
#te = TargetEncoder(target_type='multiclass', cv=5, shuffle=True, random_state=42)
#preprocessor = ColumnTransformer(transformers=[('te', te, ['sc-interaction'])],
#                                 remainder='passthrough',
#                                 verbose_feature_names_out=False)
#preprocessor.set_output(transform='pandas')

y = train.pop(TARGET_COL)
#train = preprocessor.fit_transform(train_full, y_full)
#test = preprocessor.transform(test)

model = xgb.XGBClassifier(**best_params)
model.fit(train, y)
joblib.dump(model, 'xgb_model.pkl')

# make prediction for the test data
make_prediction(model, test)

public_score = submit_prediction(COMPETITION_NAME, 'predictions_XGB_optuna.csv',
                                 f"XGB optuna 12 (0.3275286274509804)")
print(f"Public score: {public_score}")

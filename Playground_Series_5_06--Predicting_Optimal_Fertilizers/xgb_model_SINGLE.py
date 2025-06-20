import sys

import joblib
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import mapk  # noqa
from kaggle_api_functions import submit_prediction
from competition_specifics import TARGET_COL, COMPETITION_NAME, load_preprocess_data

OPTUNA_FRAC = 0.25
N_AUGMENT = 6
try:
    SERIAL_NUMBER = sys.argv[1]
except IndexError:
    SERIAL_NUMBER = 0


def make_prediction(pred_proba: NDArray) -> None:
    """ Make a prediction for the test data, with a given model.
    Args:
        pred_proba (NDArray): Averaged predicted probabilities.
    """
    submit_df = pd.read_csv('sample_submission.csv')
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
train_full, test, X_original, encoder = load_preprocess_data()
for df in [train_full, test, X_original]:
    for col in df.columns:
        df[col] = df[col].astype('category')

NUM_CLASSES = len(encoder.classes_)

y_original = X_original.pop(TARGET_COL)


params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': NUM_CLASSES,
    'tree_method': 'hist',
    'learning_rate': 0.02,
    'max_depth': 16,
    'min_child_weight': 5,
    'subsample': 0.86,
    'colsample_bytree': 0.4,
    'reg_alpha': 3,
    'reg_lambda': 1.4,
    'n_estimators': 10_000,
    'max_delta_step': 5,
    'gamma': 0.26,
    'use_label_encoder': False,
    'early_stopping_rounds': 100,
    'enable_categorical': True
}

# Train final model with best parameters
oof_preds = np.zeros((len(train_full), NUM_CLASSES))
test_fold_preds = []
mapa3_scores = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(train_full, train_full[TARGET_COL]):
    X_train_fold, X_val_fold = train_full.iloc[train_idx].copy(), train_full.iloc[val_idx].copy()

    y_train_fold = X_train_fold.pop(TARGET_COL)
    y_val_fold = X_val_fold.pop(TARGET_COL)

    weights = [1.0] * len(X_train_fold) + [N_AUGMENT] * len(X_original)
    X_train_fold = pd.concat([X_train_fold, X_original], ignore_index=True)
    y_train_fold = pd.concat([y_train_fold, y_original], ignore_index=True)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              sample_weight=weights
              )

    oof_preds[val_idx] = model.predict_proba(X_val_fold)
    top_3 = np.argsort(-oof_preds[val_idx], axis=1)[:, :3]
    actual = [[int(label)] for label in y_val_fold]
    mapa3_scores.append(mapk(actual, top_3.tolist(), k=3))

    test_fold_preds.append(model.predict_proba(test))

# save predictions for ensembling
joblib.dump({'oof_preds': oof_preds,
             'test_preds': np.mean(test_fold_preds, axis=0),
             'y_train': train_full[TARGET_COL].values
             },
             'stacking_data.pkl')

# make prediction for the test data
make_prediction(np.mean(test_fold_preds, axis=0))

public_score = submit_prediction(COMPETITION_NAME, 'predictions_XGB_single.csv',
                                 f"XGB single ({np.mean(mapa3_scores)})")
print(f"Public score: {public_score}")

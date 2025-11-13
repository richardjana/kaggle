import sys

from lightgbm import LGBMClassifier
import numpy as np
import optuna
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import loan_feature_engineering
sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_api_functions import submit_prediction

TARGET_COL = 'loan_paid_back'
COMPETITION_NAME = 'playground-series-s5e11'
N_FOLDS = 5


def load_and_prepare(file_name: str) -> pd.DataFrame:
    """ Read data from csv file and do some basic preprocessing.
    Args:
        file_name (str): Name of the csv file.
    Returns:
        pd.DataFrame: The created DataFrame.
    """
    df = pd.read_csv(file_name)
    try:  # train and test
        df.drop(columns='id', inplace=True)
    except KeyError:  # original data
        additional_cols = ['num_of_delinquencies', 'installment', 'delinquency_history',
                           'monthly_income', 'current_balance', 'total_credit_limit',
                           'public_records', 'age', 'num_of_open_accounts', 'loan_term']
        df.drop(columns=additional_cols, inplace=True)

    # convert object type columns to category type
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

    return df


def target_encode_with_original_data(df: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
    """ Use the original data for a type of target encoding.
    Args:
        df (pd.DataFrame): DataFrame to add the TE to.
        orig (pd.DataFrame): The original data.
    Returns:
        pd.DataFrame: DataFrame with added columns.
    """
    for col in [c for c in df.columns if c != TARGET_COL]:
        te_col = f"TEO_{col}"

        tmp_df = orig.groupby(col, observed=True)[TARGET_COL].mean()
        tmp_df.name = te_col
        df = df.merge(tmp_df, on=col, how='left')

        df[te_col] = df[te_col].fillna(orig[TARGET_COL].mean())

    return df


# Load dataset
X_train = load_and_prepare('train.csv')
X_test = load_and_prepare('test.csv')
orig = load_and_prepare('original.csv')

CAT_COLS = [c for c in X_train.select_dtypes(include=['category']).columns if c != TARGET_COL]
NUM_COLS = [c for c in X_train.select_dtypes(include=['number']).columns if c != TARGET_COL]

ADDITIONAL_PARAMS = {'objective': 'binary',
                     'metric': 'auc',
                     'verbosity': -1,
                     'n_jobs': -1,
                     'seed': 77,
                     'n_estimators': 10_000,
                     'early_stopping_rounds': 100
                     }

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def train_model(X_train):
    # Cross-validation
    aucs = []
    for train_idx, valid_idx in skf.split(X_train, X_train[TARGET_COL]):
        X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]

        y_train_fold = X_train_fold.pop(TARGET_COL)
        y_valid_fold = X_valid_fold.pop(TARGET_COL)

        # Train model
        model = LGBMClassifier(**ADDITIONAL_PARAMS)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_valid_fold, y_valid_fold)],
                  eval_metric='auc'
                  )

        # Predict and evaluate
        y_pred = model.predict_proba(X_valid_fold)[:, 1]
        aucs.append(roc_auc_score(y_valid_fold, y_pred))

    return model, aucs


# train plain model for comparison
model_plain, aucs_plain = train_model(X_train)
print(f"AUCs of the plain model = {aucs_plain}")

# train model on original data
model_orig, aucs_orig = train_model(orig)
print(f"AUCs of the original model = {aucs_orig}")

# predict probabilities on training set and add as feature
X_train['orig_pred'] = model_orig.predict_proba(X_train)[:, 1]

# train final model
model_augmented, aucs_augmented = train_model(X_train)
print(f"AUCs of the augmented model = {aucs_augmented}")




oof_preds = np.zeros(len(X_train))
test_fold_preds = []
for train_idx, valid_idx in skf.split(X_train, X_train[TARGET_COL]):
    X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]

    y_train_fold = X_train_fold.pop(TARGET_COL)
    y_valid_fold = X_valid_fold.pop(TARGET_COL)

    model = LGBMClassifier(**best_params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_valid_fold, y_valid_fold)],
              eval_metric='auc')

    oof_preds[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]

    test_fold_preds.append(model.predict_proba(X_test)[:, 1])


# write files for ensembling
OOF_DF = pd.DataFrame({'y_true': y_train, 'oof': oof_preds})
OOF_DF.to_csv('oof.csv', index=False)

submit_df = pd.read_csv('sample_submission.csv')
submit_df[TARGET_COL] = np.mean(np.array(test_fold_preds), axis=0).astype(float)
submit_df.to_csv('predictions_optuna.csv', columns=['id', TARGET_COL], index=False)

# submit to kaggle
public_score = submit_prediction(COMPETITION_NAME, 'predictions_optuna.csv',
                                 f"LGBM optuna ({study.best_value})")
print(f'Public score: {public_score}')

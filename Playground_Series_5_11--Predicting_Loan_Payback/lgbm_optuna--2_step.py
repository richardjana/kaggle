import sys

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

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


def train_model(X_train, X_predict=None, X_test=None):
    if X_predict is not None:
        fold_preds = []
        X_predict = X_predict.drop(columns=TARGET_COL, errors='ignore')
    if X_test is not None:
        fold_tests = []

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

        # Evaluate and predict
        aucs.append(roc_auc_score(y_valid_fold, model.predict_proba(X_valid_fold)[:, 1]))
        if X_predict is not None:
            fold_preds.append(model.predict_proba(X_predict)[:, 1])
        if X_test is not None:
            fold_tests.append(model.predict_proba(X_test)[:, 1])

    if X_predict is not None:
        prediction = np.mean(np.array(fold_preds), axis=0).astype(float)
    else:
        prediction = None
    if X_test is not None:
        testion = np.mean(np.array(fold_tests), axis=0).astype(float)
    else:
        testion = None

    return aucs, prediction, testion


def make_submission(values, file_name, message, score):
    submit_df = pd.read_csv('sample_submission.csv')
    submit_df[TARGET_COL] = values
    submit_df.to_csv(file_name, columns=['id', TARGET_COL], index=False)

    # submit to kaggle
    public_score = submit_prediction(COMPETITION_NAME, file_name,
                                    f"{message} ({score})")
    print(f'Public score: {public_score}')


# train plain model for comparison
aucs_plain, test_preds_plain, _ = train_model(X_train, X_predict=X_test, X_test=None)
print(f"AUCs of the plain model = {aucs_plain} ({np.mean(aucs_plain)})")
make_submission(test_preds_plain, 'pred_plain.csv', 'TEST plain', np.mean(aucs_plain))

# train model on original data, predict probabilities on training set and add as feature
aucs_orig, train_preds_orig, test_preds_orig = train_model(orig, X_predict=X_train, X_test=X_test)
print(f"AUCs of the original model = {aucs_orig} ({np.mean(aucs_orig)})")
X_train['orig_pred'] = train_preds_orig
X_test['orig_pred'] = test_preds_orig

# train final model
aucs_augmented, test_preds_augmented, _ = train_model(X_train, X_predict=X_test)
print(f"AUCs of the augmented model = {aucs_augmented} ({np.mean(aucs_augmented)})")
make_submission(test_preds_augmented, 'pred_augmented.csv', 'TEST augmented', np.mean(aucs_augmented))

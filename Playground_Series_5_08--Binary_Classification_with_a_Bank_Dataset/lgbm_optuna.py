import sys

from lightgbm import LGBMClassifier
import numpy as np
import optuna
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_api_functions import submit_prediction

TARGET_COL = 'y'
COMPETITION_NAME = 'playground-series-s5e8'

OPTUNA_FRAC = 0.25


def make_prediction(pipeline: Pipeline, test_df: pd.DataFrame) -> None:
    """ Make a prediction for the test data, with a given model.
    Args:
        pipeline (Pipeline): Pipeline used for the prediction.
        test_df (pd.DataFrame): DataFrame with the test data.
    """
    submit_df = pd.read_csv('sample_submission.csv')

    submit_df[TARGET_COL] = pipeline.predict(test_df)
    submit_df.to_csv('predictions_LGBM_optuna.csv',
                     columns=['id', TARGET_COL], index=False)


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, max_values=None):
        """
        cols: list of column names to encode
        max_values: dict with column names and their max values (period)
        """
        self.cols = cols
        self.max_values = max_values or {}

    def fit(self, X, y=None):
        return self  # nothing to fit

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            max_val = self.max_values.get(col)
            if max_val is None:
                raise ValueError(f"Max value (period) for column '{col}' must be specified")
            X[f'{col}_sin'] = np.sin(2 * np.pi * X[col] / max_val)
            X[f'{col}_cos'] = np.cos(2 * np.pi * X[col] / max_val)
            X.drop(columns=col, inplace=True)
        return X

    def get_feature_names_out(self, input_features=None):
        feature_names = []
        for col in self.cols:
            feature_names.append(f'{col}_sin')
            feature_names.append(f'{col}_cos')
        return np.array(feature_names)


def load_and_prepare(file_name: str) -> pd.DataFrame:
    """ Read data from csv file and do some basic preprocessing.
    Args:
        file_name (str): Name of the csv file.
    Returns:
        pd.DataFrame: The created DataFrame.
    """
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
             'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
             'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
             }

    df = pd.read_csv(file_name)
    df.drop(columns='id', inplace=True)
    df['month'] = df['month'].map(month_map)

    obj_cols = df.select_dtypes(include='object').columns
    df[obj_cols] = df[obj_cols].astype('category')

    return df


# Load dataset
X_train_full = load_and_prepare('train.csv')
X_test = load_and_prepare('test.csv')

_, X_train = train_test_split(X_train_full, test_size=OPTUNA_FRAC, random_state=42)

y_train = X_train.pop(TARGET_COL)


categorical_cols = ['job', 'marital', 'education', 'default',
                    'housing', 'loan', 'contact', 'poutcome']
skewed_cols = ['balance', 'duration', 'campaign', 'pdays', 'previous']
normal_cols = ['age']

cyclical_cols = ['day', 'month']
cyclical_max = {'month': 12, 'day': 31}
cyclical_encoder = CyclicalEncoder(cols=cyclical_cols, max_values=cyclical_max)

preprocessor = ColumnTransformer(
    transformers=[
        ('cyclical', cyclical_encoder, cyclical_cols),
        ('scale', StandardScaler(), normal_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('power', PowerTransformer(method='yeo-johnson'), skewed_cols)
    ],
    remainder='passthrough'
)


ADDITIONAL_PARAMS = {'objective': 'binary',
                     'metric': 'auc',
                     'boosting_type': 'gbdt',
                     'is_unbalance': True,
                     'verbosity': -1,
                     'n_jobs': -1,
                     'seed': 77,
                     'n_estimators': 10_000,
                     'early_stopping_rounds': 100
                     }


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Define objective function for Optuna
def objective(trial):
    params = {'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
              'num_leaves': trial.suggest_int('num_leaves', 20, 150),
              'max_depth': trial.suggest_int('max_depth', 3, 15),
              'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
              'subsample': trial.suggest_float('subsample', 0.5, 1.0),
              'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
              'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
              'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
              }
    params.update(ADDITIONAL_PARAMS)

    # Cross-validation
    aucs = []
    best_iteration_folds = []
    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # Fit preprocessor on training fold
        X_train_fold_transformed = pd.DataFrame(preprocessor.fit_transform(X_train_fold),
                                                columns=preprocessor.get_feature_names_out())
        X_valid_fold_transformed = pd.DataFrame(preprocessor.transform(X_valid_fold),
                                                columns=preprocessor.get_feature_names_out())

        # Train model
        model = LGBMClassifier(**params)
        model.fit(
            X_train_fold_transformed, y_train_fold,
            eval_set=[(X_valid_fold_transformed, y_valid_fold)],
            eval_metric='auc',
        )

        # Predict and evaluate
        pred_probas = model.predict_proba(X_valid_fold_transformed)[:, 1]
        aucs.append(roc_auc_score(y_valid_fold, pred_probas))
        best_iteration_folds.append(model.best_iteration_)

    trial.set_user_attr('n_estimators', int(np.mean(best_iteration_folds)*np.sqrt(1/OPTUNA_FRAC)))

    return np.mean(aucs)


# Create and optimize Optuna study
study = optuna.create_study(direction='maximize',
                            study_name='banking',
                            storage='sqlite:///optuna_study_lgbm.db')
study.optimize(objective, n_trials=10_000, timeout=60*60*6)



# Train final model with best parameters
y_train_full = X_train_full.pop(TARGET_COL)

best_params = study.best_params
best_params.update(ADDITIONAL_PARAMS)
best_params['n_estimators'] = study.best_trial.user_attrs.get('n_estimators')
del best_params['early_stopping_rounds']

pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LGBMClassifier(**best_params))
    ])

pipe.fit(X_train_full, y_train_full)

# make prediction for the test data
make_prediction(pipe, X_test)

public_score = submit_prediction(COMPETITION_NAME, 'predictions_LGBM_optuna.csv',
                                 f"LGBM optuna ({study.best_value})")
print(f'Public score: {public_score}')

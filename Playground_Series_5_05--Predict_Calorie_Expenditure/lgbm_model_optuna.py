from itertools import combinations
from typing import Dict, List

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_log_error

import sys
sys.path.append('../')
from kaggle_utilities import make_diagonal_plot, rmsle  # noqa

TARGET_COL = 'Calories'


def clean_data(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ Apply cleaning operations to pandas DataFrame.
    Args:
        pd_df (pd.DataFrame): The DataFrame to clean.
        drop (bool, optional): If rows should be dropped (option for training DataFrame) or not
            (test DataFrame). Defaults to True.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    pd_df.drop('id', axis=1, inplace=True)

    pd_df['Sex'] = pd_df['Sex'].map({'male': 0, 'female': 1})

    return pd_df


def add_intuitive_columns(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ Add columns to DataFrame, possibly inspired by other peoples solutions.
    Args:
        pd_df (pd.DataFrame): DataFrame to which to add new columns.
    Returns:
        pd.DataFrame: The DataFrame, with additional columns.
    """
    pd_df['BMI'] = pd_df['Weight'] / (pd_df['Height']**2) * 10000
    pd_df['BMI_Class'] = pd.cut(pd_df['BMI'],
                                bins=[0, 16.5, 18.5, 25, 30, 35, 40, 100],
                                labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)
    pd_df['BMI_zscore'] = pd_df.groupby('Sex')['BMI'].transform(zscore)

    BMR_male = 66.47 + pd_df['Weight']*13.75 + pd_df['Height']*5.003 - pd_df['Age']*6.755
    BMR_female = 655.1 + pd_df['Weight']*9.563 + pd_df['Height']*1.85 - pd_df['Age']*4.676
    pd_df['BMR'] = np.where(pd_df['Sex'] == 'male', BMR_male, BMR_female)
    pd_df['BMR_zscore'] = pd_df.groupby('Sex')['BMR'].transform(zscore)

    pd_df['Heart_rate_Zone'] = pd.cut(pd_df['Heart_Rate'], bins=[0, 90, 110, 200],
                                      labels=[0, 1, 2]).astype(int)
    pd_df['Heart_Rate_Zone_2'] = pd.cut(pd_df['Heart_Rate']/(220-pd_df['Age'])*100,
                                        bins=[0, 50, 65, 80, 85, 92, 100],
                                        labels=[0, 1, 2, 3, 4, 5]).astype(int)

    pd_df['Age_Group'] = pd.cut(pd_df['Age'], bins=[0, 20, 35, 50, 100],
                                labels=[0, 1, 2, 3]).astype(int)

    cb_male = (0.6309*pd_df['Heart_Rate'] + 0.1988*pd_df['Weight']
               + 0.2017*pd_df['Age'] - 55.0969) / 4.184 * pd_df['Duration']
    cb_female = (0.4472*pd_df['Heart_Rate'] - 0.1263*pd_df['Weight']
                 + 0.0740*pd_df['Age'] - 20.4022) / 4.184 * pd_df['Duration']
    pd_df['Calories_Burned'] = np.where(pd_df['Sex'] == 'male', cb_male, cb_female)

    for col in ['Height', 'Weight', 'Heart_Rate']:
        pd_df[f"{col}_zscore"] = pd_df.groupby('Sex')[col].transform(zscore)

    return pd_df


def generate_extra_columns(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ Generate extra feature columns from the original data, by combining columns.
    Args:
        pd_df (pd.DataFrame): The original data.
    Returns:
        pd.DataFrame: DataFrame with new columns added.
    """
    combine_cols = [col for col in pd_df.keys() if col != TARGET_COL]

    new_cols = {}

    for n in [2, 3]:
        for cols in combinations(combine_cols, n):
            col_name = '*'.join(cols)
            new_cols[col_name] = pd_df[list(cols)].prod(axis=1)

    for cols in combinations(combine_cols, 2):
        col_name = '/'.join(cols)
        new_cols[col_name] = pd_df[cols[0]] / pd_df[cols[1]]

    pd_df = pd.concat([pd_df, pd.DataFrame(new_cols, index=pd_df.index)], axis=1)

    return pd_df


def make_training_plot(history: Dict[str, List[int]], metric: str,
                       fname: str, precision: int = 2) -> None:
    """ Make plots to visualize the training progress: y-axis 1) linear scale 2) log scale.
    Args:
        history (Dict[str, List[int]]): History from model.fit.
        fname (str): File name for the plot image.
        precision (int): Number of decimals to print for the metric. Defaults to 2.
    """
    _, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
    ax.plot(np.arange(len(history['training'][metric]))+1, history['training'][metric], 'r',
            label=f"training {metric} ({min(history['training'][metric]):.{precision}f})")
    ax.plot(np.arange(len(history['valid_1'][metric]))+1, history['valid_1'][metric], 'g',
            label=f"validation {metric} ({min(history['valid_1'][metric]):.{precision}f})")
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    plt.legend(loc='best')
    plt.savefig(f"{fname}_{metric}.png", bbox_inches='tight')

    ax.set_yscale('log')
    plt.savefig(f"{fname}_{metric}_LOG.png", bbox_inches='tight')

    plt.close()


# Load dataset
dataframe = clean_data(pd.read_csv('train.csv'))
#dataframe, rest = train_test_split(dataframe, test_size=0.90)  # reduce dataset size for testing
test = clean_data(pd.read_csv('test.csv'))

dataframe = generate_extra_columns(dataframe)
test = generate_extra_columns(test)

dataframe = add_intuitive_columns(dataframe)
test = add_intuitive_columns(test)

skl_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
dataframe[TARGET_COL] = skl_transformer.transform(dataframe[[TARGET_COL]])

# Split into train/test sets
y = dataframe.pop(TARGET_COL).to_numpy()
X = dataframe.to_numpy()

# Define RMSLE function
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Define objective function for Optuna
def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 5000,
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmsle_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', 
                  callbacks=[lgb.callback.early_stopping(stopping_rounds=100),
                             lgb.callback.log_evaluation(period=0)
            ]
        )

        preds = model.predict(X_val)
        preds = skl_transformer.inverse_transform(preds.reshape(-1, 1))
        y_val_trans = skl_transformer.inverse_transform(y_val.reshape(-1, 1))

        rmsle_score = rmsle(y_val_trans, preds)
        rmsle_scores.append(rmsle_score)

    return np.mean(rmsle_scores)

# Create and optimize Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000, timeout=75600)  # Adjust timeout or n_trials as needed

# Save the study
#study_name = "lgbm_rmsle_study"
#study_storage = f"sqlite:///{study_name}.db"
#study = optuna.create_study(study_name=study_name, storage=study_storage, load_if_exists=True)

# Print best trial
print("Best trial:")
print(f"  RMSLE: {study.best_value}")
print("  Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'n_estimators': 5000
})

model = lgb.LGBMRegressor(**best_params)

eval_results = {}

model.fit(
    X, y,
    eval_set=[(X, y)],
    eval_metric='rmse',
    callbacks=[
        lgb.callback.early_stopping(stopping_rounds=100),
        lgb.callback.log_evaluation(period=50),
        lgb.callback.record_evaluation(eval_results)
    ]
)

joblib.dump(model, 'lgb_model.pkl')

make_training_plot(eval_results, 'rmse', 'training_LGBM_optuna', precision=5)

# Predict
pred_train = model.predict(X)
pred_train = skl_transformer.inverse_transform(pred_train.reshape(-1, 1))

# Evaluate
rmsle_final = rmsle(y, pred_train)
print(f'Root Mean Squared Logarithmic Error: {rmsle_final:.7f}')

# Make prediction for the test data
test_preds = model.predict(test)
test_preds = skl_transformer.inverse_transform(test_preds.reshape(-1, 1))

submit_df = pd.read_csv('sample_submission.csv')
submit_df[TARGET_COL] = test_preds
submit_df.to_csv(f"predictions_LGBM.csv", columns=['id', TARGET_COL], index=False)

y = skl_transformer.inverse_transform(y)

make_diagonal_plot(pd.DataFrame({TARGET_COL: y.ravel(), 'PREDICTION': pred_train.ravel()}),
                        pd.DataFrame({TARGET_COL: y.ravel(), 'PREDICTION': pred_train.ravel()}),
                        TARGET_COL, rmsle, 'RMSLE',
                        f"error_diagonal_LGBM_optuna.png", precision=5)

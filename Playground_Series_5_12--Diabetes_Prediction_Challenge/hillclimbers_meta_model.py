from glob import glob
import sys

from functools import partial
from hillclimbers import climb_hill
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from competition_specifics import COMPETITION_NAME, load_and_prepare, TARGET_COL
sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_api_functions import submit_prediction


# collect OOF predictions data
oof_preds_list = []
for oof_file in glob('Playground_Series_5_12--Diabetes_Prediction_Challenge_*/oof.csv'):
    df = pd.read_csv(oof_file)
    df.rename(columns={'oof': oof_file.split('/')[0].split('_')[-1]}, inplace=True)
    df.drop(columns='y_true', inplace=True)
    oof_preds_list.append(df)

# collect test predictions data
test_preds_list = []
for preds_file in glob('Playground_Series_5_12--Diabetes_Prediction_Challenge_*/predictions*.csv'):
    df = pd.read_csv(preds_file)
    df.rename(columns={'diagnosed_diabetes': preds_file.split('/')[0].split('_')[-1]}, inplace=True)
    df.drop(columns='id', inplace=True)
    test_preds_list.append(df)

X_train = load_and_prepare('train.csv')

hc_test_predictions, hc_oof_preds = climb_hill(
    train=X_train,
    oof_pred_df=pd.concat(oof_preds_list, axis=1),
    test_pred_df=pd.concat(test_preds_list, axis=1),
    target=TARGET_COL,
    objective='maximize',
    eval_metric=partial(roc_auc_score),
    negative_weights=False,  # can try True
    precision=0.01,  # recommended range 0.01 - 0.001
    plot_hill=True,
    plot_hist=True,
    return_oof_preds=True
)

# write results
OOF_DF = pd.DataFrame({'y_true': X_train[TARGET_COL], 'oof': hc_oof_preds})
OOF_DF.to_csv('oof_hillclimbers.csv', index=False)

submit_df = pd.read_csv('sample_submission.csv')
submit_df[TARGET_COL] = hc_test_predictions
submit_df.to_csv('predictions_hillclimbers.csv', columns=['id', TARGET_COL], index=False)

# submit to kaggle
n_models = len(oof_preds_list)
hc_auc = roc_auc_score(X_train[TARGET_COL], hc_oof_preds)

public_score = submit_prediction(COMPETITION_NAME, 'predictions_hillclimbers.csv',
                                 f"Hillclimbers {n_models} ({hc_auc})")
print(f"Public score: {public_score}")

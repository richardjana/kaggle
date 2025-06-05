from typing import Dict, List, Tuple
import sys

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import mapk, plot_confusion_matrix  # noqa
from kaggle_api_functions import submit_prediction
from prepare_data import load_preprocess_data

TARGET_COL = 'Fertilizer Name'

NUM_CV_SPLITS = 5

NUM_LEAVES = 31
LEARNING_RATE = 0.01
N_ESTIMATORS = 5000
L1_REGULARIZATION = 1.0
L2_REGULARIZATION = 1.0
BAGGING_FRACTION = 0.8


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

    #ax.set_yscale('log')
    plt.savefig(f"{fname}_{metric}.png", bbox_inches='tight')

    plt.close()


def make_prediction(model: lgb.LGBMClassifier, test_df: pd.DataFrame, cv_index: int | str) -> None:
    """ Make a prediction for the test data, with a given model.
    Args:
        model (lgb.LGBMRegressor): Model used for the prediction.
        test_df (pd.DataFrame): DataFrame with the test data, pre-processed.
        skl_transformer (FunctionTransformer): Transformer used to transform
            back to the original target space.
        cv_index (int | str): Index of the cross-validation fold, used in the file name.
    """
    submit_df = pd.read_csv('sample_submission.csv')
    pred_proba = np.asarray(model.predict_proba(test_df.to_numpy()))
    top_3_indices = np.argsort(-pred_proba, axis=1)[:, :3]

    top_3_labels = np.array([
        encoders[TARGET_COL].inverse_transform(sample_top3)
        for sample_top3 in top_3_indices
    ])

    joined_top3 = [' '.join(labels) for labels in top_3_labels]

    submit_df[TARGET_COL] = joined_top3
    submit_df.to_csv(f"predictions_LGBM_KFold_{cv_index}.csv",
                     columns=['id', TARGET_COL], index=False)


def lgb_map3_eval(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    """
    Custom LightGBM eval function to compute MAP@3.
    Parameters:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Flat array of predictions (num_samples * num_classes).
    Returns:
        Tuple[str, float, bool]: metric name, value, higher_is_better
    """
    # Compute top 3 predictions
    top_3 = np.argsort(-y_pred, axis=1)[:, :3]

    # Format true labels for mapk
    actual = [[label] for label in y_true]

    # Compute MAP@3
    return 'map@3', mapk(actual, top_3.tolist(), k=3), True


# Load dataset
dataframe, test, encoders = load_preprocess_data('train.csv', 'test.csv', TARGET_COL, 'LGBM')

y = dataframe.pop(TARGET_COL).to_numpy()
X = dataframe.to_numpy()
categorical_features = [dataframe.columns.get_loc(c) for c in ['Soil Type', 'Crop Type']]

if NUM_CV_SPLITS > 1:  # do cross-validation
    kfold = KFold(n_splits=NUM_CV_SPLITS, shuffle=True, random_state=42)
    scores = []
    cv_index = 0

    for train_index, val_index in kfold.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_df = dataframe.iloc[train_index].copy()
        val_df = dataframe.iloc[val_index].copy()

        cv_index += 1

        # Create the model
        model = lgb.LGBMClassifier(objective='multiclass',
                                   n_estimators=N_ESTIMATORS,
                                   learning_rate=LEARNING_RATE,
                                   num_leaves=NUM_LEAVES,
                                   subsample=BAGGING_FRACTION,
                                   subsample_freq=1,
                                   categorical_feature=categorical_features,
                                   reg_alpha=L1_REGULARIZATION,
                                   reg_lambda=L2_REGULARIZATION,
                                   class_weight='balanced',
                                   n_jobs=-1
                                   )

        eval_results = {}  # Store evaluation results

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric=lgb_map3_eval,
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=100),
                lgb.callback.log_evaluation(period=50),
                lgb.callback.record_evaluation(eval_results)
            ]
        )

        make_training_plot(eval_results, 'map@3', f"training_LGBM_{cv_index}", precision=5)

        pred_train = np.asarray(model.predict_proba(X_train))
        pred_val = np.asarray(model.predict_proba(X_val))

        y_val_decoded = encoders[TARGET_COL].inverse_transform(y_val)
        pred_val_labels = encoders[TARGET_COL].inverse_transform(pred_val.argmax(axis=1))

        plot_confusion_matrix(y_val_decoded, pred_val_labels, f"LGBM_confusion_{cv_index}.png",
                              class_names=encoders[TARGET_COL].classes_.tolist())

        # Evaluate
        top_3 = np.argsort(-pred_val, axis=1)[:, :3]
        actual = [[int(label)] for label in y_val]

        map3_score = mapk(actual, top_3.tolist(), k=3)
        print(f'MAP@3 Score: {map3_score:.7f}')
        scores.append(map3_score)

        # make prediction for the test data
        make_prediction(model, test, cv_index)

    print(f"Average cross-validation RMSLE: {np.mean(scores):.5f} ({scores})")

#else:  # train on the full data set for final prediction
cv_index = 'full'  # for plot names
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Create the model
model = lgb.LGBMClassifier(objective='multiclass',
                            n_estimators=N_ESTIMATORS,
                            learning_rate=LEARNING_RATE,
                            num_leaves=NUM_LEAVES,
                            subsample=BAGGING_FRACTION,
                            subsample_freq=1,
                            categorical_feature=categorical_features,
                            reg_alpha=L1_REGULARIZATION,
                            reg_lambda=L2_REGULARIZATION,
                            class_weight='balanced',
                            n_jobs=-1
                            )

eval_results = {}  # Store evaluation results

# Train with proper early stopping and logging callbacks
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric=lgb_map3_eval,
    callbacks=[
        lgb.callback.early_stopping(stopping_rounds=100),
        lgb.callback.log_evaluation(period=50),
        lgb.callback.record_evaluation(eval_results)
    ]
)

joblib.dump(model, 'lgb_model.pkl')

make_training_plot(eval_results, 'map@3', 'training_LGBM', precision=5)

# Predict
pred_train = np.asarray(model.predict(X_train))
pred_val = np.asarray(model.predict_proba(X_val))

y_val_decoded = encoders[TARGET_COL].inverse_transform(y_val)
pred_val_labels = encoders[TARGET_COL].inverse_transform(pred_val.argmax(axis=1))

plot_confusion_matrix(y_val_decoded, pred_val_labels, f"LGBM_confusion_{cv_index}.png",
                      class_names=encoders[TARGET_COL].classes_.tolist())

# Evaluate
top_3 = np.argsort(-pred_val, axis=1)[:, :3]
actual = [[int(label)] for label in y_val]

map3_score = mapk(actual, top_3.tolist(), k=3)
print(f'MAP@3 score: {map3_score:.7f}')

# make prediction for the test data
make_prediction(model, test, cv_index)

public_score = submit_prediction('playground-series-s5e6', f"predictions_LGBM_KFold_{cv_index}.csv",
                                 f"LGBM ({map3_score})")
print(f'Public score: {public_score:.7f}')

from itertools import combinations
import sys
from typing import Dict, List, Literal, Tuple, Union

import keras
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
import tensorflow as tf

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import min_max_scaler, make_training_plot, make_diagonal_plot, rmsle, rmsle_metric  # noqa
from prepare_calories_data import load_preprocess_data

##### hyper params for the model #####
LAYER_SIZE = 600
L2_REG = 0.01 / 50
DROP_RATE = 0.025
LEARNING_RATE_INITIAL = 1e-4
LEARNING_RATE_FINAL = 1e-6
NUM_EPOCHS = 500
BATCH_SIZE = 128
NUM_CV_SPLITS = 5
TARGET_COL = 'Calories'

LOSS_FUNCTION = tf.keras.losses.MeanSquaredError()
METRIC = 'root_mean_squared_error'


# Load dataset
log1p_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
dataframe, test = load_preprocess_data('train.csv', 'test.csv', TARGET_COL, log1p_transformer)


def make_new_model(shape: int) -> tf.keras.Model:
    """ Create a fresh model, with fresh weights and clean optimizer state.
    Args:
        shape (int): Shape (size) of the input layer.
    Returns:
        tf.keras.Model: The compiled model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(shape,)),
        tf.keras.layers.Dense(LAYER_SIZE, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
        tf.keras.layers.Dropout(DROP_RATE),
        tf.keras.layers.Dense(LAYER_SIZE, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
        tf.keras.layers.Dropout(DROP_RATE),
        tf.keras.layers.Dense(LAYER_SIZE, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
        tf.keras.layers.Dropout(DROP_RATE),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=LOSS_FUNCTION,
                  metrics=[METRIC])

    return model


def make_prediction(model: tf.keras.Model, test_df_encoded: pd.DataFrame,
                    skl_transformer: FunctionTransformer | PowerTransformer,
                    cv_index: int | str) -> None:
    """ Make a prediction for the test data, with a given model.
    Args:
        model (tf.keras.Model): Model used for the prediction.
        test_df_encoded (pd.DataFrame): DataFrame with the test data, pre-processed.
        skl_transformer (FunctionTransformer | PowerTransformer): Transformer used to transform
            back to the original target space.
        cv_index (int | str): Index of the cross-validation fold, used in the file name.
    """
    submit_df = pd.read_csv('sample_submission.csv')
    submit_df[TARGET_COL] = model.predict(test_df_encoded.to_numpy())
    submit_df[TARGET_COL] = skl_transformer.inverse_transform(submit_df[[TARGET_COL]])
    submit_df.to_csv(f"predictions_KFold_{cv_index}.csv", columns=['id', TARGET_COL], index=False)


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=round(NUM_EPOCHS**(1/2)),
    restore_best_weights=True
)


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                test_df: pd.DataFrame, cv_index: int | str) -> float:
    """ min-max-scale columns, train the model and make predictions and plots.
    Args:
        train_df (pd.DataFrame): _description_
        val_df (pd.DataFrame): _description_
        test_df (pd.DataFrame): _description_
        cv_index (int): _description_
    Returns:
        float: Score for the model between training and validation data.
    """
    # defragment DataFrames
    train_df._consolidate_inplace()
    val_df._consolidate_inplace()
    test_df._consolidate_inplace()

    # scale columns
    scale_columns = [col for col in train_df.keys() if col != TARGET_COL]
    train_df, val_df, test_df = min_max_scaler([train_df, val_df, test_df], scale_columns)

    train_target_df = pd.DataFrame({TARGET_COL: train_df.pop(TARGET_COL)})
    y_train = train_target_df.to_numpy()
    X_train = train_df.to_numpy()
    val_target_df = pd.DataFrame({TARGET_COL: val_df.pop(TARGET_COL)})
    y_val = val_target_df.to_numpy()
    X_val = val_df.to_numpy()

    model = make_new_model(shape=X_train.shape[1])
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[early_stop])

    if cv_index == 'full':
        model.save('tf_model.h5')

    make_training_plot(history.history, f"training_KFold_{cv_index}.png", precision=5)

    train_target_df[TARGET_COL] = log1p_transformer.inverse_transform(train_target_df[TARGET_COL])
    val_target_df[TARGET_COL] = log1p_transformer.inverse_transform(val_target_df[TARGET_COL])
    train_target_df['PREDICTION'] = log1p_transformer.inverse_transform(model.predict(X_train))
    val_target_df['PREDICTION'] = log1p_transformer.inverse_transform(model.predict(X_val))

    make_diagonal_plot(train_target_df, val_target_df, TARGET_COL, rmsle,
                       'RMSLE', f"error_diagonal_{cv_index}.png", precision=5)

    make_prediction(model, test_df, log1p_transformer, cv_index)

    score = rmsle(val_target_df[TARGET_COL].to_numpy(),
                  val_target_df['PREDICTION'].to_numpy())

    return score


if NUM_CV_SPLITS > 1:  # do cross-validation
    kfold = KFold(n_splits=NUM_CV_SPLITS, shuffle=True)
    scores = []
    cv_index = 0

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE_INITIAL,
        decay_steps=len(dataframe) // NUM_CV_SPLITS *
        (NUM_CV_SPLITS-1) // BATCH_SIZE,
        decay_rate=(LEARNING_RATE_FINAL /
                    LEARNING_RATE_INITIAL) ** (1 / (NUM_EPOCHS - 1)),
        staircase=True
    )

    for train_index, val_index in kfold.split(dataframe):
        train_df = dataframe.iloc[train_index].copy()
        val_df = dataframe.iloc[val_index].copy()
        test_df = test.copy()

        cv_index += 1
        cv_split_score = train_model(train_df, val_df, test_df, cv_index)
        scores.append(cv_split_score)

    print(f'Average cross-validation RMSLE: {np.mean(scores):.5f} ({scores})')

else:  # train on the full data set for final prediction
    cv_index = 'full'  # for plot names

    train_df, val_df = train_test_split(dataframe, test_size=0.1)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE_INITIAL,
        decay_steps=len(dataframe) // BATCH_SIZE,
        decay_rate=(LEARNING_RATE_FINAL /
                    LEARNING_RATE_INITIAL) ** (1 / (NUM_EPOCHS - 1)),
        staircase=True
    )

    score = train_model(train_df, val_df, test, cv_index)

    print(f"Final RMSLE score = {score}")

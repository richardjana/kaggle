import datetime
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf

import sys
sys.path.append('../')
from kaggle_utilities import make_training_plot, make_category_error_plot

##### hyper params for the model #####
layer_size = 64
L2_reg = 0.01 /10
drop_rate = 0.25
learning_rate = 0.00001
epochs = 10
cv_splits = 5
target_col = 'rainfall'


loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = 'accuracy'

stamp = datetime.datetime.timestamp(datetime.datetime.now())

def clean_data(pd_df):
    pd_df.drop('id', axis=1, inplace=True)

    # 1 missing value in 'winddirection'
    pd_df['winddirection'].fillna(pd_df['winddirection'].mean(), inplace=True)

    # replace day / winddirection with cyclic representation
    pd_df['day_sin'] = pd_df.apply(lambda row: np.sin(2*np.pi*row.day/2), axis=1)
    pd_df['day_cos'] = pd_df.apply(lambda row: np.cos(2*np.pi*row.day/2), axis=1)
    pd_df.drop('day', axis=1, inplace=True)
    pd_df['winddirection_sin'] = pd_df.apply(lambda row: np.sin(2*np.pi*row.winddirection/360), axis=1)
    pd_df['winddirection_cos'] = pd_df.apply(lambda row: np.cos(2*np.pi*row.winddirection/360), axis=1)
    pd_df.drop('winddirection', axis=1, inplace=True)

    return pd_df

##### load data #####
dataframe = clean_data(pd.read_csv('train.csv'))
#dataframe, rest = train_test_split(dataframe, test_size=0.80) # reduce dataset size for testing

def make_new_model(shape):
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(shape,)),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_reg)),
        tf.keras.layers.Dropout(drop_rate),
        tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_reg)),
        tf.keras.layers.Dropout(drop_rate),
        tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_reg)),
        tf.keras.layers.Dropout(drop_rate),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss_function,
                  metrics=[metric])

    return model

##### make predictions on the test set #####
def make_prediction(model, i):
    test = clean_data(pd.read_csv('test.csv'))
    prediction = np.argmax(tf.nn.softmax(model.predict(test.to_numpy())), axis=1).reshape(-1,)
    test = pd.read_csv('test.csv')
    test[target_col] = prediction
    test.to_csv(f"predictions_KFold_{i}.csv", columns=['id', target_col], index=False)

kfold = KFold(n_splits=cv_splits, shuffle=True)
scores = []

y_train = dataframe.pop(target_col).to_numpy()
X_train = dataframe.to_numpy()

i = 0
for train_index, val_index in kfold.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model = make_new_model(shape=X_train.shape[1])
    history = model.fit(X_train_fold, y_train_fold,
                        validation_data=(X_val_fold, y_val_fold),
                        epochs=epochs)

    model.save(f"rainfall_KFold_{i}.keras")
    make_training_plot(history.history, f"training_KFold_{i}.png")

    df_train = pd.DataFrame({target_col: y_train_fold, 'id': y_train_fold})
    df_train['PREDICTION'] = np.argmax(tf.nn.softmax(model.predict(X_train_fold)), axis=1).reshape(-1,)
    df_val = pd.DataFrame({target_col: y_val_fold, 'id': y_val_fold})
    df_val['PREDICTION'] = np.argmax(tf.nn.softmax(model.predict(X_val_fold)), axis=1).reshape(-1,)
    make_category_error_plot(df_train, target_col, f"category_error_{i}_training.png", 2)
    make_category_error_plot(df_val, target_col, f"category_error_{i}_validation.png", 2)

    make_prediction(model, i)

    i += 1
    scores.append(history.history[f"val_{metric}"][-1])

avg_score = np.mean(scores) # Calculate the average cross-validation score
print(f'Average cross-validation score: {avg_score:.4f} ({scores})')

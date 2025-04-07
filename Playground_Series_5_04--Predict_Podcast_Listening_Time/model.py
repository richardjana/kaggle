import datetime
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
#import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf

import sys
sys.path.append('../')
from kaggle_utilities import min_max_scaler, make_training_plot, make_diagonal_plot

##### hyper params for the model #####
layer_size = 64
L2_reg = 0.01 /10
drop_rate = 0.25
learning_rate = 0.00001
epochs = 1000
cv_splits = 5
target_col = 'Listening_Time_minutes'

loss_function = tf.keras.losses.MeanSquaredError()
metric = 'root_mean_squared_error'

stamp = datetime.datetime.timestamp(datetime.datetime.now())

def clean_data(pd_df, drop=True):
    pd_df.drop('id', axis=1, inplace=True)

    if drop: # drop NaN lines
        pd_df.dropna(axis=0, how='any', inplace=True)
    else: # for the test set, fill with most common / average
        pd_df['Episode_Length_minutes'].fillna(pd_df['Episode_Length_minutes'].median(), inplace=True)
        pd_df['Guest_Popularity_percentage'].fillna(pd_df['Guest_Popularity_percentage'].median(), inplace=True)
        pd_df['Number_of_Ads'].fillna(pd_df['Number_of_Ads'].median(), inplace=True)

    # make 'Episode_Title' integer column (all follow 'Episode_<number> pattern)
    pd_df['Episode_Title'] = pd_df['Episode_Title'].map(lambda et: int(et.split()[1]))

    # convert 'Publication_Day' and 'Publication_Time' columns to cyclic representation
    pd_df['Publication_Day'] = pd_df['Publication_Day'].map({'Monday': 0,
                                                             'Tuesday': 1,
                                                             'Wednesday': 2,
                                                             'Thursday': 3,
                                                             'Friday': 4,
                                                             'Saturday': 5,
                                                             'Sunday': 6})
    pd_df['day_sin'] = pd_df['Publication_Day'].map(lambda pub_day: np.sin(2*np.pi*pub_day/7))
    pd_df['day_cos'] = pd_df['Publication_Day'].map(lambda pub_day: np.cos(2*np.pi*pub_day/7))
    pd_df.drop('Publication_Day', axis=1, inplace=True)
    pd_df['Publication_Time'] = pd_df['Publication_Time'].map({'Morning': 8,
                                                               'Afternoon': 14,
                                                               'Evening': 18,
                                                               'Night': 0})
    pd_df['time_sin'] = pd_df['Publication_Time'].map(lambda pub_time: np.sin(2*np.pi*pub_time/24))
    pd_df['time_cos'] = pd_df['Publication_Time'].map(lambda pub_time: np.cos(2*np.pi*pub_time/24))
    pd_df.drop('Publication_Time', axis=1, inplace=True)

    # map to numeric columns
    pd_df['Episode_Sentiment'] = pd_df['Episode_Sentiment'].map({'Negative': -1, 'Neutral': 0, 'Positive': +1})
    # .fillna(pd_df['Episode_Sentiment'])

    # one-hot encode class columns; drop_first to avoid multicollinearity
    pd_df = pd.get_dummies(pd_df, columns=['Podcast_Name', 'Genre'], drop_first=True, dtype =int)

    return pd_df

def RMSE(arr_1, arr_2):
    return round(np.sqrt(np.sum(np.power(arr_1-arr_2, 2))/arr_1.size), 3)

##### load data #####
dataframe = clean_data(pd.read_csv('train.csv'))
#dataframe, rest = train_test_split(dataframe, test_size=0.95) # reduce dataset size for testing
test = clean_data(pd.read_csv('test.csv'), drop=False)

### scale columns (not cyclical representations, not target column) ###
scale_columns = [col for col in dataframe.keys() if (col[-4:] not in ['_sin', '_cos']) and (col != target_col)]
dataframe, test = min_max_scaler([dataframe, test], scale_columns)

def make_new_model(shape):
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(shape,)),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_reg)),
        tf.keras.layers.Dropout(drop_rate),
        tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_reg)),
        tf.keras.layers.Dropout(drop_rate),
        tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_reg)),
        tf.keras.layers.Dropout(drop_rate),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss_function,
                  metrics=[metric])

    return model

##### make predictions on the test set #####
def make_prediction(model, i):
    test_df = pd.read_csv('test.csv')
    test_df[target_col] = model.predict(test.to_numpy())
    test_df.to_csv(f"predictions_KFold_{i}.csv", columns=['id', target_col], index=False)

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
    df_train['PREDICTION'] = model.predict(X_train_fold)

    df_val = pd.DataFrame({target_col: y_val_fold, 'id': y_val_fold})
    df_val['PREDICTION'] = model.predict(X_val_fold)

    make_diagonal_plot(df_train, df_val, target_col, RMSE, 'RMSE', f"error_diagonal_{i}.png")

    make_prediction(model, i)

    i += 1
    scores.append(history.history[f"val_{metric}"][-1])

print(f'Average cross-validation RMSE: {np.mean(scores):.4f} ({scores})')

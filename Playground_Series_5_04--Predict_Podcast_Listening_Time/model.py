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
from kaggle_utilities import min_max_scaler, make_training_plot, make_diagonal_plot, RMSE

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

# TODO: Apply to all DataFrames together, so the median values are consistent?
# or rather drop from training and replace in test? (There, local or global?)
def clean_data(pd_df: pd.DataFrame, drop: bool =True) -> pd.DataFrame:
    pd_df.drop('id', axis=1, inplace=True)
   
    if drop is True: # fix erroneous values / data points ...
        pd_df.drop(pd_df[pd_df['Episode_Length_minutes']>180].index, inplace=True)
        pd_df.drop(pd_df[pd_df['Number_of_Ads']>10].index, inplace=True)
    else:
        pd_df.loc[pd_df['Episode_Length_minutes']>180, 'Episode_Length_minutes'] = pd_df['Episode_Length_minutes'].median()
        pd_df.loc[pd_df['Number_of_Ads']>10, 'Number_of_Ads'] = pd_df['Number_of_Ads'].median()

    if drop is True: # drop NaN lines
        pd_df.dropna(axis=0, how='any', inplace=True)
    else: # for the test set, fill with most common / average
        for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']:
            pd_df[col].fillna(pd_df[col].median(), inplace=True)

    # make 'Episode_Title' integer column (all follow 'Episode_<number> pattern) ...
    #pd_df['Episode_Title'] = pd_df['Episode_Title'].map(lambda et: int(et.split()[1]))
    # ... or drop it altogether?
    pd_df.drop('Episode_Title', axis=1, inplace=True)

    if drop is True: # drop / replace outliers / implausible data points
        pd_df.drop(pd_df[pd_df['Host_Popularity_percentage']<20].index, inplace=True)
        pd_df.drop(pd_df[pd_df['Host_Popularity_percentage']>100].index, inplace=True)
        pd_df.drop(pd_df[pd_df['Guest_Popularity_percentage']<0].index, inplace=True)
        pd_df.drop(pd_df[pd_df['Guest_Popularity_percentage']>100].index, inplace=True)
    else:
        col = 'Host_Popularity_percentage' # 20 <= Host_Popularity_percentage <= 100
        valid_median = pd_df[col][(pd_df[col]>=20) & (pd_df[col]<=100)].median()
        pd_df.loc[pd_df[(pd_df[col]<20) | (pd_df[col]>100)].index, col] = valid_median
        col = 'Guest_Popularity_percentage' # 0 <= Guest_Popularity_percentage <= 100
        valid_median = pd_df[col][(pd_df[col]>=0) & (pd_df[col]<=100)].median()
        pd_df.loc[pd_df[(pd_df[col]<0) | (pd_df[col]>100)].index, col] = valid_median

    return pd_df

def target_encoding(df_train, df_test):
    for col in ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Number_of_Ads', 'Episode_Sentiment']:
        groupby_df = df_train[['Listening_Time_minutes', col]].groupby([col]).mean()
        groupby_df.sort_values(by='Listening_Time_minutes', inplace=True)
        mapping_dict = groupby_df['Listening_Time_minutes'].to_dict()
        df_train[col] = df_train[col].map(mapping_dict)
        df_test[col] = df_test[col].map(mapping_dict)

    return df_train, df_test

##### load data #####
dataframe = clean_data(pd.read_csv('train.csv'))
#dataframe, rest = train_test_split(dataframe, test_size=0.95) # reduce dataset size for testing
test = clean_data(pd.read_csv('test.csv'), drop=False)

dataframe, test = target_encoding(dataframe, test)

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

    df_train = pd.DataFrame({target_col: y_train_fold})
    df_train['PREDICTION'] = model.predict(X_train_fold)

    df_val = pd.DataFrame({target_col: y_val_fold})
    df_val['PREDICTION'] = model.predict(X_val_fold)

    make_diagonal_plot(df_train, df_val, target_col, RMSE, 'RMSE', f"error_diagonal_{i}.png")

    make_prediction(model, i)

    i += 1
    scores.append(history.history[f"val_{metric}"][-1])

print(f'Average cross-validation RMSE: {np.mean(scores):.4f} ({scores})')

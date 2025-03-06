import datetime
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sys
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# python3 tf_model--cross_val.py 5 128 0.3 0.0001 250
n_layers = int(sys.argv[1]) # 5
layer_size = int(sys.argv[2]) # 128
drop_rate = float(sys.argv[3]) # 0.3
learn_rate = float(sys.argv[4]) # 0.0001
epochs = int(sys.argv[5]) # 250
cv_splits = 5

stamp = f"{n_layers}x{layer_size}_{drop_rate}_{learn_rate}_{epochs}"

def clean_data(pd_df, drop=True): # clean dataset
    pd_df.drop('id', axis=1, inplace=True)

    if drop: # drop those lines
        pd_df.dropna(axis=0, how='any', inplace=True)
    else: # for the test set, fill with most common / average
        pd_df['Brand'].fillna(pd_df['Brand'].value_counts().idxmax(), inplace=True)
        pd_df['Material'].fillna(pd_df['Material'].value_counts().idxmax(), inplace=True)
        pd_df['Size'].fillna('Medium', inplace=True)
        pd_df['Compartments'].fillna(pd_df['Compartments'].mean(), inplace=True)
        pd_df['Laptop Compartment'].fillna(0, inplace=True)
        pd_df['Waterproof'].fillna(0, inplace=True)
        pd_df['Style'].fillna(pd_df['Style'].value_counts().idxmax(), inplace=True)
        pd_df['Color'].fillna(pd_df['Color'].value_counts().idxmax(), inplace=True)
        pd_df['Weight Capacity (kg)'].fillna(pd_df['Weight Capacity (kg)'].mean(), inplace=True)

    for key, val in {'Small': 1, 'Medium': 2, 'Large': 3}.items():
        pd_df['Size'].replace(to_replace=key, value=val, inplace=True)
    for key, val in {'Yes': 1, 'No': 0}.items():
        pd_df['Laptop Compartment'].replace(to_replace=key, value=val, inplace=True)
        pd_df['Waterproof'].replace(to_replace=key, value=val, inplace=True)

    # do the encoding here
    pd_df = pd.get_dummies(pd_df, columns=['Brand', 'Material', 'Style', 'Color'], dtype=int, drop_first=True)

    return pd_df

##### load data,  split into train / validation / test #####
dataframe = clean_data(pd.concat([pd.read_csv('train.csv'), pd.read_csv('training_extra.csv')], ignore_index=True))
#dataframe, rest = train_test_split(dataframe, test_size=0.90) # reduce dataset size for testing

##### plot training history #####
def make_training_plot(history, i):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
    ax.plot(np.arange(len(history['root_mean_squared_error']))+1, history['root_mean_squared_error'], 'r', label='training RMSE')
    ax.plot(np.arange(len(history['val_root_mean_squared_error']))+1, history['val_root_mean_squared_error'], 'g', label='validation RMSE')
    ax.set_xlabel('epoch')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(f"{stamp}")
    plt.legend(loc='best')
    plt.savefig(f"training_{stamp}--KFold_{i}.png", bbox_inches='tight')
    plt.close()

def make_new_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.LayerNormalization()
    ])

    for i in range(n_layers):
        model.add(tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.Dropout(drop_rate))

    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learn_rate),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['root_mean_squared_error'])

    return model

##### make predictions on the test set #####
def make_prediction(model, i):
    test = clean_data(pd.read_csv('test.csv'), drop=False)
    #test.drop('id', axis=1, inplace=True)
    prediction = model.predict(test.to_numpy())
    test = pd.read_csv('test.csv')
    test['Price'] = prediction
    test.to_csv(f"submit_{stamp}--KFold_{i}.csv", columns=['id', 'Price'], index=False)

####### cross-validation code #######
kfold = KFold(n_splits=cv_splits, shuffle=True)
scores = []

y_train = dataframe.pop('Price').to_numpy()
X_train = dataframe.to_numpy()

i = 0
for train_index, val_index in kfold.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model = make_new_model()
    history = model.fit(X_train_fold, y_train_fold,
                        validation_data=(X_val_fold, y_val_fold),
                        epochs=epochs)

    model.save(f"backpack_{stamp}--KFold_{i}.keras")
    make_training_plot(history.history, i)
    make_prediction(model, i)
    i += 1
    scores.append(history.history['val_root_mean_squared_error'][-1])

avg_score = np.mean(scores) # Calculate the average cross-validation score
print(f'Average cross-validation score: {avg_score:.4f} ({scores})')
####### end cross-validation code #######

##### make diagonal error plot #####
#train['PREDICTION'] = model.predict(df_to_dataset(train, shuffle=False)).reshape(-1,)
#val['PREDICTION'] = model.predict(df_to_dataset(val, shuffle=False)).reshape(-1,)

'''def make_diagonal_plot(train, val):
    chart = sns.scatterplot(data=train, x='Price', y='PREDICTION', alpha=0.25)
    sns.scatterplot(data=val, x='Price', y='PREDICTION', alpha=0.25)

    min_val = min(chart.get_xlim()[0], chart.get_ylim()[0])
    max_val = max(chart.get_xlim()[1], chart.get_ylim()[1])
    chart.set_xlim([min_val, max_val])
    chart.set_ylim([min_val, max_val])
    chart.plot([min_val, max_val], [min_val, max_val], linewidth=1, color='k')

    chart.set_aspect('equal')
    chart.set_xlabel('Price')
    chart.set_ylabel('Predicted Price')

    RMSE = sklearn.metrics.root_mean_squared_error(train['Price'], train['PREDICTION'])
    labels = [f"training ({RMSE:.2f})"]
    RMSE = sklearn.metrics.root_mean_squared_error(val['Price'], val['PREDICTION'])
    labels += [f"validation ({RMSE:.2f})"]
    plt.legend(labels=labels, title='dataset (RMSE):', loc='best')

    plt.savefig(f"error_{stamp}.png", bbox_inches='tight')
    plt.close()'''

#make_diagonal_plot(train, val)

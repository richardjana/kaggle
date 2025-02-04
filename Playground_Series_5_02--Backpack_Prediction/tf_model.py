import datetime
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split

layer_size = 128
drop_rate = 0.3

stamp = datetime.datetime.timestamp(datetime.datetime.now())

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

    return pd_df

##### load data,  split into train / validation / test #####
dataframe = clean_data(pd.concat([pd.read_csv('train.csv'), pd.read_csv('training_extra.csv')], ignore_index=True))
dataframe, rest = train_test_split(dataframe, test_size=0.80) # reduce dataset size for testing
train, val = train_test_split(dataframe, test_size=0.2)
test = clean_data(pd.read_csv('test.csv'), drop=False)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    if 'Price' in df.keys():
        labels = df.pop('Price')
        df = {key: value.to_numpy()[:,tf.newaxis] for key, value in df.items()}
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(df))
    ds = ds.batch(batch_size)
    if shuffle:
        ds = ds.shuffle(buffer_size=10 * batch_size)
    ds = ds.prefetch(batch_size)
    return ds

train_ds = df_to_dataset(train)
val_ds = df_to_dataset(val)
test_ds = df_to_dataset(test, shuffle=False)

def get_normalization_layer(name, dataset):
    normalizer = layers.Normalization(axis=None) # Create a Normalization layer for the feature.
    feature_ds = dataset.map(lambda x, y: x[name]) # Prepare a Dataset that only yields the feature.
    normalizer.adapt(feature_ds) # Learn the statistics of the data.
    return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    if dtype == 'string': # Create a layer that turns strings into integer indices.
        index = layers.StringLookup(output_mode='one_hot')

    feature_ds = dataset.map(lambda x, y: x[name]) # Prepare a `tf.data.Dataset` that only yields the feature.

    index.adapt(feature_ds) # Learn the set of possible values and assign them a fixed integer index.

    return index

all_inputs = {}
encoded_features = []

numerical_columns = ['Size', 'Compartments', 'Weight Capacity (kg)', 'Laptop Compartment', 'Waterproof']
for col_name in numerical_columns:
    numeric_col = tf.keras.Input(shape=(1,), name=col_name)
    normalization_layer = get_normalization_layer(col_name, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs[col_name] = numeric_col
    encoded_features.append(encoded_numeric_col)

categorical_columns = ['Brand', 'Material', 'Style', 'Color']
for col_name in categorical_columns:
    categorical_col = tf.keras.Input(shape=(1,), name=col_name, dtype='string')
    encoding_layer = get_category_encoding_layer(name=col_name, dataset=train_ds, dtype='string')
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs[col_name] = categorical_col
    encoded_features.append(encoded_categorical_col)

all_features = tf.keras.layers.concatenate(encoded_features)

x = tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=regularizers.l2(0.01))(all_features)
x = tf.keras.layers.Dropout(drop_rate)(x)
x = tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(drop_rate)(x)
x = tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(drop_rate)(x)
x = tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(drop_rate)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(all_inputs, output)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['root_mean_squared_error'])

history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=100)

model.save(f"backpack_{stamp}.keras")

##### plot training history #####
def make_training_plot(history):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
    ax.plot(np.arange(len(history['root_mean_squared_error']))+1, history['root_mean_squared_error'], 'r', label='training RMSE')
    ax.plot(np.arange(len(history['val_root_mean_squared_error']))+1, history['val_root_mean_squared_error'], 'g', label='validation RMSE')
    ax.set_xlabel('epoch')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(f"{stamp}")
    plt.legend(loc='best')
    plt.savefig(f"training_{stamp}.png", bbox_inches='tight')
    plt.close()

make_training_plot(history.history)

##### make diagonal error plot #####
train['PREDICTION'] = model.predict(df_to_dataset(train, shuffle=False)).reshape(-1,)
val['PREDICTION'] = model.predict(df_to_dataset(val, shuffle=False)).reshape(-1,)

def make_diagonal_plot(train, val):
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
    plt.close()

make_diagonal_plot(train, val)

##### make predictions on the test set #####
test = pd.read_csv('test.csv') # read again, because I dropped the IDs above
test['Price'] = model.predict(test_ds)
test.to_csv(f"predictions_{stamp}.csv", columns=['id', 'Price'], index=False)

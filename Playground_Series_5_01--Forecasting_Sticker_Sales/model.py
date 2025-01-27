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

stamp = datetime.datetime.timestamp(datetime.datetime.now())

use_GDP = True

def clean_data(pd_df, drop=True): # clean dataset
    pd_df.drop('id', axis=1, inplace=True)

    if drop: # drop those lines
        pd_df.dropna(axis=0, how='any', inplace=True)

    # translate 'date' into a numerical value
    start = datetime.datetime.fromisoformat(pd_df['date'].min())
    time = np.empty(len(pd_df['date']))
    for i,d in enumerate(pd_df['date']):
        time[i] = (datetime.datetime.fromisoformat(d) - start).days
    pd_df['days'] = time
    pd_df.drop('date', axis=1, inplace=True)

    # replace countries with GDP values
    if use_GDP == True:
        GDP = {'Norway': 95006.33, 'Finland': 53172.67, 'Canada': 54267.67,
               'Italy': 37777.33, 'Kenya': 2089, 'Singapore': 84073}
        for key, val in GDP.items():
            pd_df['country'].replace(to_replace=key, value=val, inplace=True)

    return pd_df

##### load data,  split into train / validation / test #####
dataframe = clean_data(pd.read_csv('train.csv'))
#dataframe, rest = train_test_split(dataframe, test_size=0.80) # reduce dataset size for testing
train, val = train_test_split(dataframe, test_size=0.2)
test = clean_data(pd.read_csv('test.csv'), drop=False)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    if 'num_sold' in df.keys():
        labels = df.pop('num_sold')
        df = {key: value.to_numpy()[:,tf.newaxis] for key, value in df.items()}
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(df))
    ds = ds.batch(batch_size)
    if shuffle:
        ds = ds.shuffle(buffer_size=10 * batch_size)
    ds = ds.prefetch(batch_size)
    return ds

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

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

numerical_columns = ['days']
categorical_columns = ['country', 'store', 'product']
if use_GDP == True:
    numerical_columns += ['country']
else:
    categorical_columns += ['country']

for col_name in numerical_columns:
    numeric_col = tf.keras.Input(shape=(1,), name=col_name)
    normalization_layer = get_normalization_layer(col_name, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs[col_name] = numeric_col
    encoded_features.append(encoded_numeric_col)

for col_name in categorical_columns:
    categorical_col = tf.keras.Input(shape=(1,), name=col_name, dtype='string')
    encoding_layer = get_category_encoding_layer(name=col_name, dataset=train_ds, dtype='string')
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs[col_name] = categorical_col
    encoded_features.append(encoded_categorical_col)

all_features = tf.keras.layers.concatenate(encoded_features)

x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(all_features)
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.10)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(all_inputs, output)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), # 0.003
              loss=tf.keras.losses.MeanAbsolutePercentageError(),
              metrics=['mean_absolute_percentage_error']) # root_mean_squared_log_error

history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=500)#,
          #callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 30))])

model.save(f"insurance_{stamp}.keras")


## TEST LEARNING RATE AND STUFF CODE COULD GO HERE


##### plot training history #####
def make_training_plot(history):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
    ax.plot(np.arange(len(history['mean_absolute_percentage_error']))+1, history['mean_absolute_percentage_error'], 'r', label='training MAPE')
    ax.plot(np.arange(len(history['val_mean_absolute_percentage_error']))+1, history['val_mean_absolute_percentage_error'], 'g', label='validation MAPE')
    ax.set_xlabel('epoch')
    ax.set_ylabel('Mean Average Percentage Error')
    ax.set_title(f"{stamp}")
    plt.legend(loc='best')
    plt.savefig(f"training_{stamp}.png", bbox_inches='tight')
    plt.close()

make_training_plot(history.history)

##### make predictions on the test set #####
predictions = model.predict(test_ds)
np.savetxt(f"submit_{stamp}.csv", np.append(pd.read_csv('test.csv').to_numpy()[:, 0].reshape(-1, 1), predictions, axis=1), fmt='%i', delimiter=',', header='id,Premium Amount', comments='')

##### make diagonal error plot #####
def make_diagonal_plot(training_value, training_prediction, validation_value, validation_prediction):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)

    MAPE = sklearn.metrics.mean_absolute_percentage_error(training_value, training_prediction)
    ax.scatter(training_value, training_prediction, alpha=0.25, label=f"training ({MAPE:.2f})")
    MAPE = sklearn.metrics.mean_absolute_percentage_error(validation_value, validation_prediction)
    ax.scatter(validation_value, validation_prediction, alpha=0.25, label=f"validation ({MAPE:.2f})")

    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_val, max_val], [min_val, max_val], linewidth=1, color='k')
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

    ax.set_aspect('equal')
    ax.legend(loc='best', title='dataset (MAPE):')
    ax.set_xlabel('actual')
    ax.set_ylabel('predicted')
    fig.savefig(f"error_{stamp}.png")

train_ds = df_to_dataset(train, shuffle=False, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
make_diagonal_plot(train['num_sold'].to_numpy().reshape(-1, 1),
                   model.predict(train_ds),
                   val['num_sold'].to_numpy().reshape(-1, 1),
                   model.predict(val_ds))

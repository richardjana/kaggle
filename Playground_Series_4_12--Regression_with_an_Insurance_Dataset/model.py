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
#print(tf.__version__)

from tensorflow import feature_column
from tensorflow.keras import layers
#from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split

stamp = datetime.datetime.timestamp(datetime.datetime.now())

def clean_data(pd_df): # clean dataset
def clean_data(pd_df, drop=True): # clean dataset
    pd_df.drop('id', axis=1, inplace=True)
    # drop those lines
    if drop:
        pd_df.dropna(axis=0, how='any', inplace=True)
    # OR replace unknown non-numerical values (or should I remove them?)
    for key in ['Marital Status', 'Occupation', 'Customer Feedback']:
        pd_df[key] = pd_df[key].fillna('UNKNOWN')

    # setting numerical column NaNs to median value for the column
    # consider dropping some values here
    for key in ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration']:
        pd_df[key] = pd_df[key].fillna(pd_df[key].median())

    # translate 'Policy Start Date' into a numerical value
    delta = np.empty(len(pd_df['Policy Start Date']))
    now = datetime.datetime.now()
    for i,psd in enumerate(pd_df['Policy Start Date']):
        d = now - datetime.datetime.fromisoformat(psd)
        delta[i] = d.days
    pd_df['Time Since Start'] = delta
    pd_df.drop('Policy Start Date', axis=1, inplace=True)

    # turn ordered categorical columns into numerical ones
    pd_df.dropna(axis=0, how='any', inplace=True)
    for key, val in {'High School': 12, "Bachelor's": 15, "Master's": 17, 'PhD': 22}.items():
        pd_df['Education Level'].replace(to_replace=key, value=val, inplace=True)
    for key, val in {'Daily': 1, 'Weekly': 7, 'Monthly': 30, 'Rarely': 90}.items():
        pd_df['Exercise Frequency'].replace(to_replace=key, value=val, inplace=True)
    for key, val in {'Poor': 0, 'Average': 1, 'Good': 2, 'UNKNOWN': 1}.items():
        pd_df['Customer Feedback'].replace(to_replace=key, value=val, inplace=True)
    for key, val in {'Rural': 0, 'Suburban': 1, 'Urban': 2}.items():
        pd_df['Location'].replace(to_replace=key, value=val, inplace=True)
    for key, val in {'Basic': 0, 'Comprehensive': 1, 'Premium': 2}.items():
        pd_df['Policy Type'].replace(to_replace=key, value=val, inplace=True)

    return pd_df

##### load data,  split into train / validation / test #####
dataframe = clean_data(pd.read_csv('train.csv'))
dataframe, rest = train_test_split(dataframe, test_size=0.95) # reduce dataset size for testing
train, val = train_test_split(dataframe, test_size=0.2)
test = clean_data(pd.read_csv('test.csv'))
test = clean_data(pd.read_csv('test.csv'), drop=False)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    if 'Premium Amount' in df.keys():
        labels = df.pop('Premium Amount')
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

# try different encoding strategies: one-hot for unrelated categories, integer for sorted ones
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    if dtype == 'string': # Create a layer that turns strings into integer indices.
        index = layers.StringLookup(max_tokens=max_tokens)
    else: # Otherwise, create a layer that turns integer values into integer indices.
        index = layers.IntegerLookup(max_tokens=max_tokens)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Encode the integer indices.
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))

all_inputs = {}
encoded_features = []

# Numerical features.
for col_name in ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration', 'Time Since Start', 'Education Level', 'Location', 'Customer Feedback', 'Policy Type', 'Exercise Frequency']:
    numeric_col = tf.keras.Input(shape=(1,), name=col_name)
    normalization_layer = get_normalization_layer(col_name, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs[col_name] = numeric_col
    encoded_features.append(encoded_numeric_col)

for col_name in ['Gender', 'Marital Status', 'Occupation', 'Smoking Status', 'Property Type']:
    categorical_col = tf.keras.Input(shape=(1,), name=col_name, dtype='string')
    encoding_layer = get_category_encoding_layer(name=col_name, dataset=train_ds, dtype='string', max_tokens=10)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs[col_name] = categorical_col
    encoded_features.append(encoded_categorical_col)

all_features = tf.keras.layers.concatenate(encoded_features)


x = tf.keras.layers.Dense(128, activation='relu')(all_features)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.10)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(all_inputs, output)

model.compile(optimizer=keras.optimizers.Adam(), # learning_rate=1e-3
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['root_mean_squared_error']) # root_mean_squared_log_error

history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=100,
          callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 30))])

model.save(f"insurance_{stamp}.keras")
# Recreate the exact same model purely from the file:
#model = keras.models.load_model("path_to_my_model.keras")

##### estimate optimal learning rate for optimizer #####
# (run small dataset for many epochs with LR scheduler, make plot loss vs. LR)
def plot_LR(history):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), tight_layout=True)

    ax[0].plot(np.arange(len(history['root_mean_squared_error']))+1, history['root_mean_squared_error'], c='tab:blue', label='RMSE')
    ax[0].set_ylabel('RMSE', color='tab:blue')
    ax[0].tick_params(axis='y', labelcolor='tab:blue')
    twin = ax[0].twinx()
    twin.plot(np.arange(len(history['root_mean_squared_error']))+1, history['learning_rate'], '--k', label='Learning rate')
    twin.set_ylabel('Learning rate', color='k')
    ax[0].set_xlabel('Epoch')

    ax[1].plot(history['learning_rate'], history['loss'], 'k')
    m = np.argmin(history['loss'])
    ax[1].plot(history['learning_rate'][m], history['loss'][m], 'or')
    ax[1].text(history['learning_rate'][m], history['loss'][m], f"{history['learning_rate'][m]}", ha='center', va='top')
    ax[1].set_xlabel('Learning rate')
    ax[1].set_ylabel('Loss')
    ax[1].set_xscale('log')

    plt.savefig(f"LR_test_{stamp}.png", bbox_inches='tight')
    plt.close()

plot_LR(history.history)

##### save training history to logfile (final epoch) #####
with open('log.txt', 'a') as LOG:
    LOG.write(f"{stamp}: {history.history['root_mean_squared_error'][-1]}, {history.history['val_root_mean_squared_error'][-1]}\n")

##### plot training history #####
def make_training_plot(history):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
    ax.plot(np.arange(len(history['root_mean_squared_error']))+1, history['root_mean_squared_error'], 'r', label='training RMSE')
    ax.plot(np.arange(len(history['val_root_mean_squared_error']))+1, history['val_root_mean_squared_error'], 'g', label='validation RMSE')
    ax.set_xlabel('epoch')
    ax.set_ylabel('Root Mean Squared Error (€)')
    ax.set_title(f"{stamp}")
    plt.legend(loc='best')
    plt.savefig(f"training_{stamp}.png", bbox_inches='tight')
    plt.close()

make_training_plot(history.history)

##### make predictions on the test set #####
predictions = model.predict(test_ds)
np.savetxt(f"submit_{stamp}.csv", np.append(pd.read_csv('test.csv').to_numpy()[:, 0].reshape(-1, 1), predictions, axis=1), fmt='%i', delimiter=',', header='id,Premium Amount')
np.savetxt(f"submit_{stamp}.csv", np.append(pd.read_csv('test.csv').to_numpy()[:, 0].reshape(-1, 1), predictions, axis=1), fmt='%i', delimiter=',', header='id,Premium Amount', comments='')

##### make diagonal error plot #####
def RMSE(arr_1, arr_2):
    return round(np.sqrt(np.sum(np.power(arr_1-arr_2, 2))/arr_1.size), 3)

def make_diagonal_plot(training_value, training_prediction, validation_value, validation_prediction):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)

    ax.scatter(training_value, training_prediction, alpha=0.5, label=f"training ({RMSE(training_value, training_prediction):.2f})")
    ax.scatter(validation_value, validation_prediction, alpha=0.5, label=f"validation ({RMSE(validation_value, validation_prediction):.2f})")

    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_val, max_val], [min_val, max_val], linewidth=1, color='k')
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

    ax.set_aspect('equal')
    ax.legend(loc='best', title='dataset (RMSE):')
    ax.set_xlabel('actual')
    ax.set_ylabel('predicted')
    fig.savefig(f"error_{stamp}.png")

train_ds = df_to_dataset(train, shuffle=False, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
make_diagonal_plot(train['Premium Amount'].to_numpy().reshape(-1, 1),
                   model.predict(train_ds),
                   val['Premium Amount'].to_numpy().reshape(-1, 1),
                   model.predict(val_ds))

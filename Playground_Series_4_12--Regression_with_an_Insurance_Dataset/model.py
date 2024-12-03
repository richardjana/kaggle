import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
print(tf.__version__)

from tensorflow import feature_column
#from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def clean_data(pd_df): # clean dataset
# replace unknown non-numerical values (or should I remove them?)
    for key in ['Marital Status', 'Occupation', 'Customer Feedback']:
        pd_df[key] = pd_df[key].fillna('UNKNOWN')

    # setting numerical column NaNs to median value for the column
    for key in ['Annual Income', 'Number of Dependents', 'Health Score', 'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration']:
        pd_df[key] = pd_df[key].fillna(pd_df[key].median())

    return pd_df

##### load data,  split into train / validation / test #####
dataframe = clean_data(pd.read_csv("train.csv"))
train, val = train_test_split(dataframe, test_size=0.2)
test = clean_data(pd.read_csv("test.csv"))

#print(train.sample(n=5))
#print(test.sample(n=5))

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    if 'Premium Amount' in dataframe.keys():
        labels = dataframe.pop('Premium Amount')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 64
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


# In TensorFlow 1, you usually perform feature preprocessing with the tf.feature_column API. In TensorFlow 2, you can do this directly with Keras preprocessing layers.
# https://www.tensorflow.org/guide/migrate/migrating_feature_columns
feature_columns = []

# numeric cols
for col_name in ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration']:
    feature_columns.append(feature_column.numeric_column(col_name))

# b) integer / ordinal encoding -> education level, customer feedback, exercise frequency

# embedding columns
for col_name in ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location', 'Policy Type', 'Policy Start Date', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 'Property Type']:
    column = feature_column.categorical_column_with_vocabulary_list(col_name, dataframe[col_name].unique())
    embedded = feature_column.embedding_column(column, dimension=8)
    feature_columns.append(embedded)



feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(.1),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)




exit()

# make prediction
predictions = model(x_train[:1]).numpy()
print('predictions untrained', predictions)

# The tf.nn.softmax function converts these logits to probabilities for each class
print('logits', tf.nn.softmax(predictions).numpy())

# Define a loss function for training using losses.SparseCategoricalCrossentropy
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3
print('loss untrained', loss_fn(y_train[:1], predictions).numpy())

# Before you start training, configure and compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train and evaluate your model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
# or
# history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# The Model.evaluate method checks the model's performance, usually on a validation set or test set.
model.evaluate(x_test,  y_test, verbose=2)

# If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print('model', model(x_test[:1]), np.sum(model(x_test[:1]).numpy()))
print('probability_model', probability_model(x_test[:1]), np.sum(probability_model(x_test[:1]).numpy()))

# The returned history object holds a record of the loss values and metric values during training
print('history', history.history)


#model.save("path_to_my_model.keras")
#del model
# Recreate the exact same model purely from the file:
#model = keras.models.load_model("path_to_my_model.keras")

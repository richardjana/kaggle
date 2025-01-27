import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_val = x_train[-10000:] # Reserve 10,000 samples for validation
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Build a tf.keras.Sequential model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

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

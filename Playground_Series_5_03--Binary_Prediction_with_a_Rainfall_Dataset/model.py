import datetime
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf

##### hyper params for the model #####
layer_size = 64
L2_reg = 0.01 /10
drop_rate = 0.25
learning_rate = 0.00001
epochs = 10
cv_splits = 5

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = 'accuracy'

stamp = datetime.datetime.timestamp(datetime.datetime.now())

def clean_data(pd_df):
    pd_df.drop('id', axis=1, inplace=True)

    # replace day with cyclic representation
    pd_df['day_sin'] = pd_df.apply(lambda row: np.sin(2*np.pi*row.day/2), axis=1)
    pd_df['day_cos'] = pd_df.apply(lambda row: np.cos(2*np.pi*row.day/2), axis=1)
    pd_df.drop('day', axis=1, inplace=True)

    return pd_df

##### load data,  split into train / validation / test #####
dataframe = clean_data(pd.read_csv('train.csv'))
#dataframe, rest = train_test_split(dataframe, test_size=0.80) # reduce dataset size for testing
train, val = train_test_split(dataframe, test_size=0.2)

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

##### plot training history #####
def make_training_plot(history, i):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
    ax.plot(np.arange(len(history[metric]))+1, history[metric], 'r', label=f"training {metric}")
    ax.plot(np.arange(len(history[f"val_{metric}"]))+1, history[f"val_{metric}"], 'g', label=f"validation {metric}")
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    ax.set_title(f"{stamp}")
    plt.legend(loc='best')
    plt.savefig(f"training_KFold_{i}.png", bbox_inches='tight')
    plt.close()

##### make predictions on the test set #####
def make_prediction(model, i):
    test = clean_data(pd.read_csv('test.csv'))
    prediction = np.argmax(tf.nn.softmax(model.predict(test.to_numpy())), axis=1).reshape(-1,)
    test = pd.read_csv('test.csv')
    test['rainfall'] = prediction
    test.to_csv(f"predictions_KFold_{i}.csv", columns=['id', 'rainfall'], index=False)

def make_category_error_plot(pd_df, i, name, n=2):
    h_map = np.zeros((n, n))
    counts = pd_df[['id', 'rainfall', 'PREDICTION']].groupby(['rainfall', 'PREDICTION'], as_index=False).count().values.tolist()
    for r, p, c in counts:
        h_map[p, r] = c

    h_map /= np.sum(h_map)

    cmap = sns.color_palette('rocket', as_cmap=True)
    chart = sns.heatmap(h_map, cmap=cmap, square=True, linewidths=.5, cbar_kws={'shrink': .5})

    for y in range(n):
        for x in range(n):
            txt = plt.text(x + 0.5, y + 0.5, f"{h_map[y, x]:.3f}", ha='center', va='center')
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w'), PathEffects.withStroke(linewidth=1, foreground='k')])

    chart.invert_yaxis()
    chart.set_xlabel('Label')
    chart.set_ylabel('Predicted label')

    plt.savefig(f"category_error_{i}_{name}.png", bbox_inches='tight')
    plt.close()

kfold = KFold(n_splits=cv_splits, shuffle=True)
scores = []

y_train = dataframe.pop('rainfall').to_numpy()
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
    make_training_plot(history.history, i)

    df_train = pd.DataFrame({'rainfall': y_train_fold, 'id': y_train_fold})
    df_train['PREDICTION'] = np.argmax(tf.nn.softmax(model.predict(X_train_fold)), axis=1).reshape(-1,)
    df_val = pd.DataFrame({'rainfall': y_val_fold, 'id': y_val_fold})
    df_val['PREDICTION'] = np.argmax(tf.nn.softmax(model.predict(X_val_fold)), axis=1).reshape(-1,)
    make_category_error_plot(df_train, i, 'training', n=2)
    make_category_error_plot(df_val, i, 'validation', n=2)

    make_prediction(model, i)

    i += 1
    scores.append(history.history[f"val_{metric}"][-1])

avg_score = np.mean(scores) # Calculate the average cross-validation score
print(f'Average cross-validation score: {avg_score:.4f} ({scores})')

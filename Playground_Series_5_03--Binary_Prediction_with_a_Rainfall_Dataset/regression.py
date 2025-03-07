import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append('../')
from kaggle_utilities import make_category_error_plot

target_col = 'rainfall'

def clean_data(pd_df): # clean dataset
    pd_df.drop('id', axis=1, inplace=True)

    # replace day / winddirection with cyclic representation
    pd_df['day_sin'] = pd_df.apply(lambda row: np.sin(2*np.pi*row.day/2), axis=1)
    pd_df['day_cos'] = pd_df.apply(lambda row: np.cos(2*np.pi*row.day/2), axis=1)
    pd_df.drop('day', axis=1, inplace=True)
    pd_df['winddirection_sin'] = pd_df.apply(lambda row: np.sin(2*np.pi*row.winddirection/360), axis=1)
    pd_df['winddirection_cos'] = pd_df.apply(lambda row: np.cos(2*np.pi*row.winddirection/360), axis=1)
    pd_df.drop('winddirection', axis=1, inplace=True)

    return pd_df

##### load data,  split into train / validation / test #####
dataframe = clean_data(pd.read_csv('train.csv'))
dataframe, rest = train_test_split(dataframe, test_size=0.80) # reduce dataset size for testing
train, val = train_test_split(dataframe, test_size=0.2)
test = clean_data(pd.read_csv('test.csv'))

def fit_linear_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)

    return model

def x_y_pd_dataframe(pd_df):
    y = pd_df[target_col].to_numpy()
    x_df = pd_df.drop(target_col, axis=1)
    X = x_df.to_numpy()

    return X, y

# train linear model
X, y = x_y_pd_dataframe(train)
model = fit_linear_model(X, y)

# make predictions on train / test set
train['PREDICTION'] = model.predict(X)
print("Training accuracy:", accuracy_score(train[target_col], train['PREDICTION']))
print('Training F1:', f1_score(train[target_col], train['PREDICTION']))
X, y = x_y_pd_dataframe(val)
val['PREDICTION'] = model.predict(X)
print("Validation accuracy:", accuracy_score(val[target_col], val['PREDICTION']))
print('Validation F1:', f1_score(val[target_col], val['PREDICTION']))

train.rename(columns={'pressure': 'id'}, inplace=True)
val.rename(columns={'pressure': 'id'}, inplace=True)
make_category_error_plot(train, target_col, 'category_error_training.png', 2)
make_category_error_plot(val, target_col, 'category_error_validation.png', 2)

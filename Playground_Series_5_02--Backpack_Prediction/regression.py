import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def clean_data(pd_df, drop=True): # clean dataset
    pd_df.drop('id', axis=1, inplace=True)

    if drop: # drop NaN lines
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

    # transform labels into numbers: Size, Laptop Compartment, Waterproof
    for key, val in {'Small': 1, 'Medium': 2, 'Large': 3}.items():
        pd_df['Size'].replace(to_replace=key, value=val, inplace=True)
    for key, val in {'Yes': 1, 'No': 0}.items():
        pd_df['Laptop Compartment'].replace(to_replace=key, value=val, inplace=True)
        pd_df['Waterproof'].replace(to_replace=key, value=val, inplace=True)

    # one-hot encode: Brand, Material, Style, Color
    pd_df = pd.get_dummies(pd_df, columns=['Brand', 'Material', 'Style', 'Color'], drop_first=True, dtype =int) # drop_first to avoid multicollinearity

    return pd_df

##### load data,  split into train / validation / test #####
dataframe = clean_data(pd.read_csv('train.csv'))
dataframe, rest = train_test_split(dataframe, test_size=0.80) # reduce dataset size for testing
train, val = train_test_split(dataframe, test_size=0.2)
test = clean_data(pd.read_csv('test.csv'), drop=False)

def fit_rmse_linear_model(X, y):
    # Ensure X is a 2D array
    #X = np.asarray(X)
    #y = np.asarray(y).squeeze()

    # Add bias term to X (intercept for the model)
    X = np.column_stack((np.ones(X.shape[0]), X))

    # Define the RMSE loss function
    def rmse_loss(beta, X, y):
        return sklearn.metrics.root_mean_squared_error(y, X @ beta)

    # Initial guess for parameters
    init_params = np.ones(X.shape[1]) * np.mean(y)

    # Minimize the MAPE loss
    result = minimize(rmse_loss, init_params, args=(X, y), method='L-BFGS-B')

    # Extract optimized parameters
    beta_opt = result.x

    # Create and return an sklearn LinearRegression model
    model = LinearRegression()
    model.coef_ = beta_opt[1:]
    model.intercept_ = beta_opt[0]

    return model

def x_y_pd_dataframe(pd_df):
    y = pd_df['Price'].to_numpy()
    x_df = pd_df.drop('Price', axis=1)
    X = x_df.to_numpy()

    return X, y

# train linear model
X, y = x_y_pd_dataframe(train)
model = fit_rmse_linear_model(X, y)

# make predictions on train / test set
#X, y = x_y_pd_dataframe(train)
train['PREDICTION'] = model.predict(X)
X, y = x_y_pd_dataframe(val)
val['PREDICTION'] = model.predict(X)

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

    labels = [f"training ({sklearn.metrics.root_mean_squared_error(train['Price'], train['PREDICTION']):.2f})"]
    labels += [f"validation ({sklearn.metrics.root_mean_squared_error(val['Price'], val['PREDICTION']):.2f})"]
    plt.legend(labels=labels, title='dataset (RMSE):', loc='best')

    plt.savefig(f"error_linear.png", bbox_inches='tight')
    plt.close()

make_diagonal_plot(train, val)

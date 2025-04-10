import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import holidays
from sklearn.linear_model import Ridge

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

# make RMSE model!

def fit_mape_linear_model(X, y):
    # Ensure X is a 2D array
    X = np.asarray(X)
    y = np.asarray(y).squeeze()

    # Add bias term to X
    X = np.column_stack((np.ones(X.shape[0]), X))

    # Define the MAPE loss function
    def mape_loss(beta, X, y):
        y_pred = X @ beta
        return np.mean(np.abs((y - y_pred) / y)) * 100

    # Initial guess for parameters
    init_params = np.zeros(X.shape[1])

    # Minimize the MAPE loss
    result = minimize(mape_loss, init_params, args=(X, y), method='L-BFGS-B')

    # Extract optimized parameters
    beta_opt = result.x

    # Create and return an sklearn LinearRegression model
    model = LinearRegression()
    model.coef_ = beta_opt[1:]
    model.intercept_ = beta_opt[0]
    return model

train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])
train = train.dropna().reset_index(drop=True)
df = pd.concat([train, test], sort=False).reset_index(drop=True)

# Useful Columns
df['year'] = df['date'].dt.year
df['n_day'] = (df['date'] - df['date'].min()).dt.days
df['weekday'] = df['date'].dt.weekday
df['day_of_year'] = df['date'].dt.dayofyear

# Generate Wave Columns
wave_columns = []
# subtract leap dates
df.loc[df['date'] > dt.datetime(2012, 2, 29), 'n_day'] -= 1
df.loc[df['date'] > dt.datetime(2016, 2, 29), 'n_day'] -= 1

for i in range(1, 10):

    df[f'wave_sin{i}'] = np.sin(np.pi * i * df['n_day'] / 365)
    df[f'wave_cos{i}'] = np.cos(np.pi * i * df['n_day'] / 365)
    wave_columns.append(f'wave_sin{i}')
    wave_columns.append(f'wave_cos{i}')

# Near Holiday
df['near_holiday'] = 0
for country in df['country'].unique():
    days = [day for day in holidays.CountryHoliday(country, years=df['year'].unique())]
    for day in days:
        df.loc[(df.country == country) & (df['date'].dt.date < day + dt.timedelta(days=10)) & (df['date'].dt.date > day - dt.timedelta(days=10)), 'near_holiday'] = 1

# GDP Factor
import requests
def get_gdp_per_capita(country,year):
    alpha3 = {'Canada': 'CAN', 'Finland': 'FIN',
              'Italy': 'ITA', 'Kenya': 'KEN',
              'Norway': 'NOR', 'Singapore': 'SGP'}
    url="https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json".format(alpha3[country],year)
    response = requests.get(url).json()
    return response[1][0]['value']

gdp = np.array([[get_gdp_per_capita(country, year) for year in df['year'].unique()] for country in df['country'].unique()])
gdp_df = pd.DataFrame(gdp, columns=df['year'].unique(), index=df['country'].unique())
for year in df['year'].unique():
    for country in df['country'].unique():
        df.loc[(df['year'] == year) & (df['country'] == country), 'gdp'] = gdp_df.loc[country, year]

df['gdp_factor'] =  (-17643.346899+85.42355636*df['gdp']) / 365

fig, axs = plt.subplots(1, 2, figsize=(20, 5))
grouped_data = df.groupby(['date', 'year', 'country'])['num_sold'].sum().reset_index()
total_per_day = df.groupby('year')['num_sold'].sum().reset_index()
grouped_data = grouped_data.merge(total_per_day, on=['year'], suffixes=['', '_total']).reset_index()
grouped_data = grouped_data.merge(df[['date', 'country', 'gdp_factor']], on=['date', 'country'])

for country in df['country'].unique():
    country_data = grouped_data[((grouped_data['country'] == country) & (grouped_data['date'] < dt.datetime(2017, 1, 1)))]
    axs[0].plot(country_data['date'], country_data['num_sold'] / country_data['num_sold_total'], '-', label=country)
    axs[0].plot(country_data['date'], country_data['gdp_factor'] / country_data['num_sold_total'], 'b--')
axs[0].set_title('Amt Sold Per Country')
axs[0].legend()

for country in df['country'].unique():
    country_data = grouped_data[((grouped_data['country'] == country) & (grouped_data['date'] < dt.datetime(2017, 1, 1)))]
    axs[1].plot(country_data['date'], country_data['num_sold'] / country_data['gdp_factor'], '-', label=country)
axs[1].set_title('Amt Sold Per Country normalized by GDP factor')
axs[1].legend()

df['ratio'] = df['gdp_factor']
df['total'] = df['num_sold'] / df['ratio']

plt.tight_layout()
plt.show()

# Store Factor
df_no_can_ken = df[~df['country'].isin(['Canada', 'Kenya'])]

fig, axs = plt.subplots(1, 2, figsize=(10, 6))
store_data = df_no_can_ken.groupby(['date', 'store'])['num_sold'].sum().reset_index()
total_per_day = df_no_can_ken.groupby('date')['num_sold'].sum().reset_index()
store_data = store_data.merge(total_per_day, on=['date'], suffixes=['', '_total'])

# Calculate store factor
store_data['store_factor'] = store_data['num_sold'] / store_data['num_sold_total']
store_df = store_data.groupby('store')['store_factor'].mean().reset_index()
store_data.drop('store_factor', axis=1, inplace=True)
store_data = store_data.merge(store_df, on=['store'])
print(f"Store factor sum is {store_df['store_factor'].sum()}")

# Merge store factor into df
df = df.drop('store_factor', axis=1, errors='ignore')
df = df.merge(store_df, on=['store'])
df['ratio'] = df['store_factor']

for store in df['store'].unique():
    data = store_data[store_data['store'] == store]
    axs[0].plot(data['date'], data['num_sold'] / data['num_sold_total'], '.', label=f'Store {store}')
axs[0].set_title('Relative Amt Sold Per Store')
axs[0].legend()

# Normalize by current ratio
for store in df['store'].unique():
    data = store_data[store_data['store'] == store]
    axs[1].plot(data['date'], data['num_sold'] /  data['num_sold_total'] /  data['store_factor'], '.', label=f'Store {store}')
axs[1].set_title('Rel Amt Sold Per Store normalized by store factor')
axs[1].legend()

plt.tight_layout()
plt.show()

# Product Factor
df['ratio'] = df['store_factor'] * df['gdp_factor']
df['total'] = df['num_sold'] / df['ratio']

df_no_can_ken = df[~df['country'].isin(['Canada', 'Kenya'])]
total_per_day = df_no_can_ken.groupby('date')['total'].sum().reset_index()
df_no_can_ken = df_no_can_ken.merge(total_per_day, on=['date'], suffixes=['', '_per_day'])
df_no_can_ken['total_perc_per_day'] = df_no_can_ken['total'] / df_no_can_ken['total_per_day']
fig, axs = plt.subplots(1, 2, figsize=(20, 5))

# fit wave columns to each product
df['product_factor'] = None
for product in df['product'].unique():

    df_product = df_no_can_ken[((df_no_can_ken['product'] == product) & (df_no_can_ken['date'] < dt.datetime(2017, 1, 1)))].groupby('date')
    X = df_product[wave_columns].mean()
    y = df_product['total_perc_per_day'].sum()

    model = fit_mape_linear_model(X, y)
    df.loc[df['product'] == product, 'product_factor'] = model.predict(df[df['product'] == product][wave_columns])

    axs[0].plot(df_product['date'].unique().index, y, '-', label=product)
    axs[0].plot(df_product['date'].unique().index, model.predict(X), 'b--')
axs[0].set_title('Amt Sold Per Product')
axs[0].legend()

# Visualize the result
df_no_can_ken = df[~df['country'].isin(['Canada', 'Kenya'])]
total_per_day = df_no_can_ken.groupby('date')['total'].sum().reset_index()
df_no_can_ken = df_no_can_ken.merge(total_per_day, on=['date'], suffixes=['', '_per_day'])
df_no_can_ken['total_perc_per_day'] = df_no_can_ken['total'] / df_no_can_ken['total_per_day']
for product in df['product'].unique():
    df_product = df_no_can_ken[((df_no_can_ken['product'] == product) & (df_no_can_ken['date'] < dt.datetime(2017, 1, 1)))].groupby('date')
    y = df_product['total_perc_per_day'].sum()
    product_factor = df_product['product_factor'].mean()
    axs[1].plot(df_product['date'].unique().index, y / product_factor, '-', label=product)
axs[1].set_title('Amt Sold Per Product normalized by Product factor')
axs[1].legend()

df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor']
df['total'] = df['num_sold'] / df['ratio']

# Day of Week Factor
df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor']
df['total'] = df['num_sold'] / df['ratio']

df_no_can_ken_hol = df[~((df['country'].isin(['Canada', 'Kenya'])) | (df['near_holiday']))]

mean_per_weekday = df_no_can_ken_hol.groupby('weekday')['total'].mean().reset_index()
mean_mon_thur = mean_per_weekday[mean_per_weekday['weekday'] < 4]['total'].mean()
ratio_per_weekday = mean_per_weekday.copy()
ratio_per_weekday['day_of_week_factor'] = ratio_per_weekday['total'] / mean_mon_thur
ratio_per_weekday = ratio_per_weekday.drop('total', axis=1)

df = df.drop('day_of_week_factor', axis=1, errors='ignore')
df = df.merge(ratio_per_weekday, on='weekday')

grouped_data = df_no_can_ken_hol.groupby(['date'])['total'].mean().reset_index()
grouped_data = grouped_data[grouped_data['date'] < dt.datetime(2017, 1, 1)]

fig, axs = plt.subplots(1, 2, figsize=(20, 5))

axs[0].plot(grouped_data['date'], grouped_data['total'], '-')
axs[0].set_title('Mean Total Per Day')

df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor']
df['total'] = df['num_sold'] / df['ratio']

# Visualize the result
df_no_can_ken_hol = df[~((df['country'].isin(['Canada', 'Kenya'])) | (df['near_holiday']))]

grouped_data = df_no_can_ken_hol.groupby(['date'])['total'].mean().reset_index()
grouped_data = grouped_data[grouped_data['date'] < dt.datetime(2017, 1, 1)]

axs[1].plot(grouped_data['date'], grouped_data['total'], '-')
axs[1].set_title('Mean Total Per Day normalized by weekday factor')

# Sincos factor
df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor']
df['total'] = df['num_sold'] / df['ratio']

df_no_can_ken_hol = df[~((df['country'].isin(['Canada', 'Kenya'])) | (df['near_holiday']))]
grouped_data = df_no_can_ken_hol[df_no_can_ken_hol['date'] < dt.datetime(2017, 1, 1)].groupby(['date'])
X = grouped_data[wave_columns].mean()
y = grouped_data['total'].mean()

model = fit_mape_linear_model(X, y)

df['sincos_factor'] = model.predict(df[wave_columns])

fig, axs = plt.subplots(1, 2, figsize=(20, 5))

axs[0].plot(grouped_data['date'].unique().index, y, '-')
axs[0].set_title('Mean Total Per Day')
axs[0].plot(grouped_data['date'].unique().index, model.predict(X), 'r--')


df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor'] * df['sincos_factor']
df['total'] = df['num_sold'] / df['ratio']

# Visualize the result
df_no_can_ken_hol = df[~((df['country'].isin(['Canada', 'Kenya'])) | (df['near_holiday']))]
grouped_data = df_no_can_ken_hol.groupby(['date'])['total'].mean().reset_index()

axs[1].plot(grouped_data['date'], grouped_data['total'], '-')
axs[1].set_title('Mean Total Per Day normalized by sincos factor')


# Trend Factor
# This factor is only meant to make calculating the subsequent factors easier.
# I attempted to include this factor in the final product for one of my submissions. Doing this while also removing the 1.06 factor (see const_factor under the "Prediction and Submission" section) resulted in my best public LB score (0.04422). However, as expected, this was very overfit (private LB ~0.054).
df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor'] * df['sincos_factor']
df['total'] = df['num_sold'] / df['ratio']

grouped_data = df.groupby(['date', 'n_day'])['total'].mean().reset_index()



fig, axs = plt.subplots(1, 2, figsize=(20, 5))
axs[0].plot(grouped_data['date'], grouped_data['total'], '-')

train = grouped_data[(grouped_data['date'] < dt.datetime(2017, 1, 1)) & (grouped_data['date'] > dt.datetime(2012, 12, 31))]
X = train['n_day'].to_numpy().reshape(-1, 1)
y = train['total']

model = Ridge(alpha=0.1)
model.fit(X, y)

df['trend_factor'] = model.predict(df['n_day'].to_numpy().reshape(-1, 1))
df.loc[df['date'] < dt.datetime(2013, 1, 1), 'trend_factor'] = 1
axs[0].plot(grouped_data['date'], model.predict(grouped_data['n_day'].to_numpy().reshape(-1, 1)), 'r--')
axs[0].set_title('Mean Total Over Time Uncorrected')

df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor'] * df['sincos_factor'] * df['trend_factor']
df['total'] = df['num_sold'] / df['ratio']

# Visualize the result
grouped_data = df.groupby(['date', 'n_day'])['total'].mean().reset_index()
axs[1].plot(grouped_data['date'], grouped_data['total'], '-')
axs[1].set_title('Mean Total Over Time Corrected by Trend Factor')

# Country Factor
df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor'] * df['sincos_factor'] * df['trend_factor']
df['total'] = df['num_sold'] / df['ratio']

grouped_data = df[df['product'] == "Kaggle"].groupby(['date', 'country'])['total'].sum().reset_index()
total_per_day = df[df['product'] == "Kaggle"].groupby('date')['total'].sum().reset_index()
grouped_data = grouped_data.merge(total_per_day, on=['date'], suffixes=['', '_per_day'])
grouped_data = grouped_data[grouped_data['date'] < dt.datetime(2017, 1, 1)]

df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor'] * df['sincos_factor'] * df['trend_factor']
df['total'] = df['num_sold'] / df['ratio']

fig, axs = plt.subplots(1, 2, figsize=(20, 5))

for country in df['country'].unique():
    country_data = grouped_data[grouped_data['country'] == country]
    axs[0].plot(country_data['date'], country_data['total'] / country_data['total_per_day'], '-', label=country)
axs[0].set_title('Mean Total Per Day Per Country')
axs[0].legend()

country_factor = df[(df['product'] == 'Kaggle')].groupby('country').total.sum().rename('country_factor')
country_factor = country_factor / country_factor.median()
df = df.drop('country_factor', axis=1, errors='ignore')
df = df.merge(country_factor, on='country')
df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor'] * df['sincos_factor'] * df['country_factor']
df['total'] = df['num_sold'] / df['ratio']

# Visualize the result
grouped_data = df[df['product'] == "Kaggle"].groupby(['date', 'country'])['total'].sum().reset_index()
total_per_day = df[df['product'] == "Kaggle"].groupby('date')['total'].sum().reset_index()
grouped_data = grouped_data.merge(total_per_day, on=['date'], suffixes=['', '_per_day'])
grouped_data = grouped_data[grouped_data['date'] < dt.datetime(2017, 1, 1)]

for country in df['country'].unique():
    country_data = grouped_data[grouped_data['country'] == country]
    axs[1].plot(country_data['date'], country_data['total'] / country_data['total_per_day'], '-', label=country)
axs[1].set_title('Mean Total Per Day Per Country normalized by country factor')
axs[1].legend()

# Holiday factor
# My handling of the following two factors (holiday factor and New Years factor) is inspired by JZ's first place solution to a previous competition.
# Define the years and countries
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
countries = df['country'].unique()
# Initialize an empty list to hold DataFrames
dfs = []
# Generate holidays for each country and year
for year in years:
    for country in countries:
        for date, holiday_name in sorted(holidays.CountryHoliday(country, years=year).items()):

            df_0 = pd.DataFrame({"date": [date], "country": [
                country]})
            dfs.append(df_0)

# Concatenate all the DataFrames
df_holidays = pd.concat(dfs, ignore_index=True)
# Convert 'date' column to datetime
df_holidays['date'] = pd.to_datetime(df_holidays['date'])
df_holidays['tmp'] = 1

for column in df.columns:
    if 'holiday_' in column:
        df = df.drop(column, axis=1)

# holidays
holidays_columns = []
for i in range(0, 10):
    column = 'holiday_{}'.format(i)
    shifted = df_holidays.rename(columns={'tmp': column})
    shifted['date'] = shifted['date'] + dt.timedelta(days=i)
    df = pd.merge(df, shifted, on=['country', 'date'], how='left')
    df[column].fillna(0, inplace=True)
    df[column] = df[column]
    holidays_columns.append(column)

fig, axs = plt.subplots(1, 2, figsize=(20, 5))

df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor'] * df['sincos_factor'] * df['country_factor'] * df['trend_factor']
df['total'] = df['num_sold'] / df['ratio']

axs[0].plot(df['date'], df['total'], '-')
axs[0].set_title('Total Over Time')

# fit linear model to total using holidays

train = df[(df['date'] > dt.datetime(2012, 12, 31)) & (df['date'] < dt.datetime(2017, 1, 1))]
X = train[holidays_columns]
y = train['total']
model = fit_mape_linear_model(X, y)

df['holiday_factor'] = model.predict(df[holidays_columns])

df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor'] * df['sincos_factor'] * df['country_factor'] * df['holiday_factor'] * df['trend_factor']
df['total'] = df['num_sold'] / df['ratio']

axs[1].plot(df['date'], df['total'], '-')
axs[1].set_title('Total Over Time normalized by holiday factor')

# New Years Factor
new_years_columns = []
for day in range(25, 32):
    column = 'day_12_{}'.format(day)
    df[column] = ((df['date'].dt.month == 12) & (df['date'].dt.day == day)).astype(float)
    new_years_columns.append(column)
for day in range(1, 11):
    column = 'day_1_{}'.format(day)
    df[column] = ((df['date'].dt.month == 1) & (df['date'].dt.day  == day)).astype(float)
    new_years_columns.append(column)

fig, axs = plt.subplots(1, 2, figsize=(20, 5))

df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor'] * df['sincos_factor'] * df['country_factor'] * df['holiday_factor'] * df['trend_factor']
df['total'] = df['num_sold'] / df['ratio']

axs[0].plot(df['date'], df['total'], '-')
axs[0].set_title('Total Over Time')

train = df[(df['date'] > dt.datetime(2012, 12, 31)) & (df['date'] < dt.datetime(2017, 1, 1))]
X = train[new_years_columns]
y = train['total']
model = fit_mape_linear_model(X, y)

df['new_years_factor'] = model.predict(df[new_years_columns])

df['ratio'] = df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor'] * df['sincos_factor'] * df['country_factor'] * df['holiday_factor'] * df['new_years_factor'] * df['trend_factor']
df['total'] = df['num_sold'] / df['ratio']

axs[1].plot(df['date'], df['total'], '-')
axs[1].set_title('Total Over Time normalized by new years factor')

# Prediction and Submission


df['ratio'] = df['country_factor'] * df['store_factor'] * df['gdp_factor'] * df['product_factor'] * df['day_of_week_factor'] * df['sincos_factor'] * df['holiday_factor'] * df['new_years_factor']
df['total'] = df['num_sold'] / df['ratio']

# Multiplying the predictions by 1.06 seems to improve the public LB score.
# I'm not entirely sure why, but I suspect it has to do with the fact that the model is off by ~6% by 2017 (as shown in the right plot of the sincos section above).
const_factor = df['total'].median() * 1.06

df['prediction'] = df['ratio'] * const_factor

fig, ax = plt.subplots(1, 1, figsize=(20, 5))
ax.plot(df['date'], df['total'])

from sklearn.metrics import mean_absolute_percentage_error

mape_train = mean_absolute_percentage_error(df[(df['date'] < dt.datetime(2017, 1, 1)) & (~pd.isna(df.num_sold))].num_sold, df[(df['date'] < dt.datetime(2017, 1, 1)) & (~pd.isna(df.num_sold))].prediction)

print(f'{mape_train=}')

df['prediction'] = np.round(df['prediction']).astype(float).astype(int)

submission = df[df['date'] >= dt.datetime(2017, 1, 1)][['id', 'prediction']].rename(columns={'prediction': 'num_sold'})

# timestampt submission filename
submission_filename = dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_submission.csv'

submission.to_csv(f"{submission_filename}", index=False)

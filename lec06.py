import pandas as pd
import numpy as np

from coreforecast.scalers import boxcox, boxcox_lambda
from statsforecast import StatsForecast
from statsforecast.arima import arima_string
from statsforecast.models import AutoARIMA, SeasonalNaive
from statsmodels.tsa.stattools import kpss


# Read data
df = pd.read_csv('lec06_df.csv', parse_dates=['ds'])
a10 = pd.read_csv('lec06_a10.csv', parse_dates=['ds'])
seasonal_ts = pd.read_csv('lec06_seasonal_ts.csv', parse_dates=['ds'])

# Differencing
df_diff = df.copy()
df_diff['y'] = df_diff.groupby('unique_id')['y'].diff()
df_diff['unique_id'] = df_diff['unique_id'] + '_diff'
df_combined = pd.concat([df, df_diff])
StatsForecast.plot(df=df_combined)

# KPSS test
stat, p_value, lags, crit = kpss(df['y'], regression='c')
print("p-value:", p_value)
stat, p_value, lags, crit = kpss(df['y'].diff().dropna(), regression='c')
print("p-value:", p_value)

# Boxcox
df= a10.copy()
lambda_guerrero = boxcox_lambda(df['y'].values, season_length=12, method='guerrero')
print(lambda_guerrero)
df_boxcox = df.copy()
df_boxcox['y'] = boxcox(df['y'].values, lmbda=lambda_guerrero)
df_boxcox['unique_id'] = 'Boxcox_guerrero transformed'
df_combined = pd.concat([df, df_boxcox])
StatsForecast.plot(df=df_combined)

# Forecasting with ARIMA and seasonality
sf = StatsForecast(models=[AutoARIMA(season_length=12, alias='SARIMA'), SeasonalNaive(season_length=12)], freq='MS')
fc = sf.fit_predict(df=seasonal_ts, h=12)
StatsForecast.plot(df=seasonal_ts, forecasts_df=fc)
print(sf.models)
print(sf.uids)
print(arima_string(sf.fitted_[0][0].model_))
print(arima_string(sf.fitted_[1][0].model_))
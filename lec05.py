import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Holt, HoltWinters, SimpleExponentialSmoothing

# Load data
medical_ts = pd.read_csv('lec05_medical_ts.csv', parse_dates=['ds'])
df = pd.read_csv('lec05_df.csv', parse_dates=['ds'])
seasonal_addmult = pd.read_csv('lec05_seasonal_addmult.csv', parse_dates=['ds'])

# Exponential Smoothing and different alphas
data = [28, 30, 24, 25, 24, 27, 30, 26, 30, 22]
dates = pd.date_range(start='2010', periods=len(data), freq='YS')
df = pd.DataFrame({'ds': dates, 'y': data, 'unique_id': 'SES data'})
models = [
    SimpleExponentialSmoothing(alpha=0.2, alias='SES_0.2'),
    SimpleExponentialSmoothing(alpha=0.4, alias='SES_0.4'),
    SimpleExponentialSmoothing(alpha=0.6, alias='SES_0.6')
]
sf = StatsForecast(models=models, freq='YS')
fc_df = sf.forecast(df=df, h=2, fitted=True)
fitted_fc = sf.forecast_fitted_values()
sf.plot(df=df.head(1), forecasts_df=pd.concat([fitted_fc.dropna(), fc_df]))

# Single Exponential Smoothing
sf = StatsForecast(models=[AutoETS(model='ANN', prediction_intervals=None)], freq='MS')
fc_df = sf.forecast(df=medical_ts, h=12, level=[80, 95])
sf.plot(df=medical_ts, forecasts_df=fc_df, level=[80], unique_ids=['T30'])

# Double Exponential Smoothing
sf = StatsForecast(models=[Holt()], freq='MS')
fc_df = sf.fit_predict(df=df, h=24, level=[80])
sf.plot(df=df, forecasts_df=fc_df, level=[80])

# Dampened trend for Double Expoential Smoothing
sf = StatsForecast(models=[Holt(), AutoETS(model='AAN', damped=True, phi=0.8, alias='Holt dampened')], freq='MS')
fc_df = sf.forecast(df=df, h=12, fitted=True, level=[80])
sf.plot(df=df, forecasts_df=fc_df, max_insample_length=12)

# Holt-Winters forecast with additive and multiplicative seasonality
models=[HoltWinters(season_length=12, error_type='A', alias='Additive Holt-Winters'), HoltWinters(season_length=12, error_type='M', alias='Multiplicative Holt-Winters')]
sf = StatsForecast(models=models, freq='MS')
fc_df = sf.forecast(df=seasonal_addmult, h=36, fitted=True, level=[80, 95])
sf.plot(df=seasonal_addmult, forecasts_df=fc_df)

# AutoETS forecasts and prediction intervals
sf = StatsForecast(models=[AutoETS(season_length=12, damped=True)], freq='MS')
fc_df = sf.fit_predict(df=seasonal_addmult, h=12, level=[80, 95])
sf.plot(df=seasonal_addmult, forecasts_df=fc_df, level=[80])

print(sf.models)
print(sf.uids)
print(sf.fitted_[0][0].model_['method'])



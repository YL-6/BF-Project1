import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from statsforecast import StatsForecast
from statsforecast.models import (
    HistoricAverage,
    Naive,
    RandomWalkWithDrift,
    SeasonalNaive,
    SeasonalWindowAverage,
    SklearnModel,
    WindowAverage,
)
from functools import partial
from utilsforecast.feature_engineering import pipeline, trend, time_features
from sklearn.linear_model import LinearRegression


# Import data
ts_data = pd.read_csv('lec03_ts_data.csv', parse_dates=['ds'])
medical_ts = pd.read_csv('lec03_medical_ts.csv', parse_dates=['ds'])
seasonal_ts = pd.read_csv('lec03_seasonal_ts.csv', parse_dates=['ds'])

# Forecast example, always change name or model or run several models in one go
# Also change dataframe if wanted
name = 'Mean model forecasts'
plot_engine = 'plotly'  # Could also be 'matplotlib'
models = [HistoricAverage()]  # Change/add models here

sf = StatsForecast(models=models, freq='MS')
forecast_df = sf.forecast(df=medical_ts, id_col='product_id', target_col='sales', h=12)  # Depending on dataframe change column names
fig = sf.plot(medical_ts, forecast_df, id_col='product_id', target_col='sales', engine=plot_engine)
fig

# TSLM
def monthly_dummies(times):
    dummies = pd.get_dummies(times.month, prefix='month', drop_first=True, dtype=int)
    return dummies

train_features, future_features = pipeline(
    seasonal_ts,
    features=[
        trend,        
        partial(time_features, features=[monthly_dummies]),
    ],
    freq='MS',
    h=12,
)
sf = StatsForecast(
    models=[SklearnModel(LinearRegression())],
    freq='MS',
)
predictions = sf.forecast(
    df=train_features,
    h=12,
    X_df=future_features
)
fig = sf.plot(seasonal_ts, predictions, engine='matplotlib')
fig

# Additional: Forecast over time
name = 'Updated forecasts over time'
medical_t29 = medical_ts[medical_ts['product_id']=='T29'].copy()
models = [Naive(), WindowAverage(window_size=12)]
sf = StatsForecast(models=models, freq='MS')
cv_df = sf.cross_validation(df=medical_t29, step_size=6, n_windows=9, h=6, id_col='product_id', target_col='sales')
plt.figure(figsize=(15, 6))
plt.plot(medical_t29['ds'], medical_t29['sales'], label='Original Sales', color='black')
for cutoff in cv_df['cutoff'].unique():
    df_cut = cv_df[cv_df['cutoff'] == cutoff]
    plt.plot(df_cut['ds'], df_cut['Naive'], label='Naive Forecast' if cutoff == cv_df['cutoff'].unique()[0] else "", 
             color='red', alpha=0.4)
    plt.plot(df_cut['ds'], df_cut['WindowAverage'], label='WindowAverage Forecast' if cutoff == cv_df['cutoff'].unique()[0] else "", 
             color='blue', alpha=0.4)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(name)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
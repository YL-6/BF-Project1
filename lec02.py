import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import month_plot, plot_acf
from tsfeatures import tsfeatures, acf_features

from src.decomposition import STL, decomposition_plot

# Import data
ts = pd.read_csv('lec02_ts.csv', parse_dates=['ds'])
ts_data = pd.read_csv('lec02_ts_data.csv', parse_dates=['ds'])
product14 = pd.read_csv('lec02_product14.csv', parse_dates=['ds'])

# STL decomposition
stl = STL(seasonality_period=12, model='additive', lo_frac=0.4)
decomposed_ts = stl.fit(ts.set_index('ds').drop(columns='unique_id'))
fig = decomposition_plot(ts.index, decomposed_ts.observed['y'], decomposed_ts.seasonal['y'], decomposed_ts.trend['y'], decomposed_ts.resid['y'], width=1000, height=1000)
fig.show(renderer="svg")

# Seasonal Strength + Residual Variability
def quantify_fs_rv(ts: pd.DataFrame, seasonality_period: int, lo_frac: float):
    stl = STL(seasonality_period=seasonality_period, model='additive', lo_frac=lo_frac)
    decomposed_ts = stl.fit(ts.set_index('ds').drop(columns='unique_id'))
    fs = max(0, 1 - np.var(decomposed_ts.resid['y']) / np.var(decomposed_ts.resid['y'] + decomposed_ts.seasonal['y']))
    rv = np.std(decomposed_ts.resid['y']) / ts['y'].mean()
    return pd.Series({'strength_seasonality': round(fs, 2), 'residual_variability': round(rv, 2)})

ts_data.groupby('unique_id').apply(lambda x: quantify_fs_rv(x, seasonality_period=12, lo_frac=0.4))

# Seasonality Plot 2
month_plot(ts.set_index('ds')['y'])
plt.show()

# Seasonality Plot 1
ts['Year'] = ts['ds'].dt.year
ts['Month'] = ts['ds'].dt.month
pivot = ts.pivot_table(index='Month', columns='Year', values='y')
pivot.plot(figsize=(10, 6), marker='o')
plt.title('Seasonality Plot (Monthly Values per Year)')
plt.xlabel('Month')
plt.ylabel('Value')
plt.xticks(range(1, 13))
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Autocorrelation Plot
plot_acf(product14['y'], lags=15)  # You can change lags as needed
plt.title("Autocorrelation Plot")
plt.show()

# Autocorrelation Quantification  
features = tsfeatures(product14, features=[acf_features], freq=12)
features[features.select_dtypes(include='number').columns] = features.select_dtypes(include='number').round(2)
features
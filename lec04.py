import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import HistoricAverage, Naive, RandomWalkWithDrift, SeasonalNaive
import utilsforecast.losses as ufl
from utilsforecast.evaluation import evaluate

# Read data
seasonal_ts = pd.read_csv('lec04_seasonal_ts.csv', parse_dates=['ds'])

# Perform cross-validation
models = [Naive(), RandomWalkWithDrift(), SeasonalNaive(season_length=12)]
sf = StatsForecast(models=models, freq='MS')
cv_df = sf.cross_validation(df=seasonal_ts, step_size=6, n_windows=5, h=6)

# Calculate metrics overall per time-series
metrics = [ufl.mae, ufl.mse, ufl.mape]
metrics_df = evaluate(df=cv_df.drop(columns={'cutoff'}),  # remove 'cutoff' as otherwise it thinks of it as a model
                   train_df=seasonal_ts, metrics=metrics)

# Calculate metrics per time series and horizon
cv_df['horizon'] = (cv_df['ds'].dt.year - cv_df['cutoff'].dt.year) * 12 + (cv_df['ds'].dt.month - cv_df['cutoff'].dt.month)
evaluations = cv_df.groupby('horizon').apply(
    lambda x: evaluate(df=x.drop(columns={'cutoff'}), train_df=seasonal_ts, metrics=[ufl.mae, ufl.mse, ufl.mape]), include_groups=False)
evaluations.reset_index().drop('level_1', axis=1).sort_values(['unique_id', 'metric', 'horizon'])

# Prediction Intervals
models = [Naive(), HistoricAverage(), SeasonalNaive(season_length=12)]
sf = StatsForecast(models=models, freq='MS')
fc_df = sf.forecast(df=seasonal_ts, h=12, level=[80, 95])
StatsForecast.plot(df=seasonal_ts, forecasts_df=fc_df, level=[80], unique_ids=['Newspaper and book retailing'], max_insample_length=48)
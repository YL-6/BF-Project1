import pandas as pd
import numpy as np
import warnings
import utilsforecast.losses as ufl

from statsforecast import StatsForecast
from statsforecast.models import (
    Naive,
    RandomWalkWithDrift,
    #ETS,
    SeasonalNaive,
    #TSLM,
    AutoARIMA,
    HistoricAverage
)
from statsmodels.tsa.seasonal import seasonal_decompose

def get_user_inputs():
    # Get file name
    file_name = input("Enter the file name (with extension, e.g., data.csv): ").strip()

    # Get list of time series IDs to forecast
    product_classes_input = input("Enter time series IDs to forecast, separated by commas: ").strip()
    if len(product_classes_input) > 5 | len(product_classes_input) == 0:
        raise ValueError("Incorrect many product classes. Input 1 up to 5 product classes.")
    product_classes_to_forecast = [id_.strip() for id_ in product_classes_input.split(",") if id_.strip()]

    # Get metric and validate
    metric = input("Enter the metric ('mape' or 'mse'): ").strip().lower()
    if metric not in ['mape', 'mse']:
        raise ValueError("Invalid metric. Must be 'mape' or 'mse'.")

    return file_name, product_classes_to_forecast, metric


def load_selected_timeseries(file_name: str, product_classes: list[str]) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    df.dropna(subset=["product_class", "Month", "sales_volume"], inplace=True)
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df.dropna(subset=["Month"], inplace=True)
    df = df[df["product_class"].isin(product_classes)]
    df = df.sort_values("Month")

    formatted_df = df.rename(columns={
        "product_class": "unique_id",
        "Month": "ds",
        "sales_volume": "y"
    })[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    return formatted_df


def check_seasonality_strength(ts_df: pd.DataFrame, freq: int = 12, threshold: float = 0.5) -> str:
    """
    Determines if a time series DataFrame has high or low seasonality.

    Args:
        ts_df (pd.DataFrame): DataFrame with columns ['unique_id', 'ds', 'y'].
        freq (int): Number of periods in a season.
        threshold (float): Strength threshold to label "high" or "low".

    Returns:
        str: "high" or "low" seasonality.
    """
    ts = ts_df.set_index("ds")["y"]
    inferred_freq = pd.infer_freq(ts.index)

    if inferred_freq is None or len(ts.dropna()) < 2 * freq:
        return "low"

    ts = ts.asfreq(inferred_freq)

    try:
        decomposition = seasonal_decompose(ts, model='additive', period=freq, extrapolate_trend='freq')
        resid = decomposition.resid.dropna()
        seasonal = decomposition.seasonal.loc[resid.index]
        var_resid = np.var(resid)
        var_combined = np.var(resid + seasonal)
        if var_combined == 0:
            return "low"
        strength = 1 - (var_resid / var_combined)
        return "high" if strength >= threshold else "low"
    except Exception:
        return "low"


def select_category(ts_df: pd.DataFrame) -> str:
    """
    Assigns a forecast category to a time series DataFrame.

    Args:
        ts_df (pd.DataFrame): DataFrame with columns ['unique_id', 'ds', 'y']

    Returns:
        str: Forecast category
    """
    y = ts_df["y"]
    if len(y) < 48:
        return "Category 1"
    elif check_seasonality_strength(ts_df) == "high" and (y == 0).any():
        return "Category 2"
    elif check_seasonality_strength(ts_df) == "high" and (y > 0).all():
        return "Category 3"
    elif check_seasonality_strength(ts_df) == "low" and (y == 0).any():
        return "Category 4"
    else:
        return "Category 5"


def run_cross_validation(ts_df: pd.DataFrame, metric: str, min_length: int = 48) -> pd.DataFrame:
    """
    Runs time-series-specific cross-validation based on category-defined models.

    Args:
        ts_df (pd.DataFrame): DataFrame containing all series with columns ['unique_id', 'ds', 'y']
        metric (str): 'mape' or 'mse'
        min_length (int): Minimum number of observations required per series

    Returns:
        pd.DataFrame: All CV forecasts with actuals and metadata
    """
    all_cv_dfs = []

    # Group by unique_id (series)
    for unique_id, group_df in ts_df.groupby("unique_id"):
        if len(group_df) < min_length:
            print(f"Skipping {unique_id}: not enough data (<{min_length} periods)")
            continue

        if metric == 'mape' and (group_df['y'] == 0).any():
            warnings.warn(f"MAPE may be unreliable for {unique_id}: contains zero values.")

        category = select_category(group_df)
        model_names = categories[category]
        models = [model_registry[name] for name in model_names]

        print(f"\nRunning CV for {unique_id} in {category} using models: {model_names}")

        sf = StatsForecast(models=models, freq='MS')
        cv_df = sf.cross_validation(
            df=group_df,
            h=12,
            step_size=6,
            n_windows=4,
        )
        cv_df['unique_id'] = unique_id  # keep consistent naming
        all_cv_dfs.append(cv_df)

    if all_cv_dfs:
        return pd.concat(all_cv_dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # Empty if no valid series


def calculate_accuracy_metrics(cv_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # Drop cutoff if present, it's irrelevant for metric calculation
    df_for_eval = cv_df.drop(columns=['cutoff'], errors='ignore')

    exclude_cols = {'ds', 'y', 'product_class'}
    model_cols = [col for col in df_for_eval.columns if col not in exclude_cols and col != 'unique_id']

    metric_func = ufl.mape if metric == 'mape' else ufl.mse

    # Calculate metric per series, per model
    accuracy_df = metric_func(
        df=df_for_eval,
        models=model_cols,
        id_col='unique_id',
        target_col='y'
    )

    # accuracy_df has index 'unique_id' and columns = model names, each cell = metric for that series-model pair
    return accuracy_df


def select_best_models(accuracy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best model (lowest error) per time series based on accuracy dataframe.

    Returns a DataFrame with:
    - 'unique_id'
    - 'best_model'
    - 'best_metric' (accuracy value)
    """
    # Drop any non-numeric columns (like 'unique_id' if it's in the index)
    numeric_accuracy_df = accuracy_df.select_dtypes(include='number')

    # Find the model with the minimum error per row
    best_models = numeric_accuracy_df.idxmin(axis=1)
    best_scores = numeric_accuracy_df.min(axis=1)

    # Construct the output DataFrame
    best_df = pd.DataFrame({
        'unique_id': accuracy_df.index,
        'best_model': best_models,
        'best_metric': best_scores
    })

    return best_df



def forecast_best_models(
    ts_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    horizon: int = 12,
    output_path: str = "final_forecasts.csv"
) -> pd.DataFrame:
    """
    Forecasts using the best model for each time series, prints accuracy comparison,
    and saves all forecasts to a single CSV.

    Parameters:
    - ts_df: Full original time series data with columns ['unique_id', 'ds', 'y'].
    - best_models_df: DataFrame with ['unique_id', 'best_model', 'best_metric'].
    - accuracy_df: DataFrame with model performance per series.
    - horizon: Forecast horizon (default=12).
    - output_path: File path for the CSV output.
    """

    all_forecasts = []

    for _, row in best_models_df.iterrows():
        series_id = row['unique_id']
        best_model = row['best_model']
        best_score = row['best_metric']

        # Catch KeyError if Naive wasn't used for this series
        try:
            naive_score = accuracy_df.loc[series_id, 'Naive']
        except KeyError:
            naive_score = None

        # Filter series
        series_df = ts_df[ts_df['unique_id'] == series_id].copy()

        # Build and run model
        model = model_registry[best_model]
        sf = StatsForecast(models=[model], freq='MS')

        forecast_df = sf.forecast(df=series_df[['unique_id', 'ds', 'y']], h=horizon)

        forecast_df = forecast_df.rename(columns={
            best_model: f"forecast_{series_id}_{best_model}"
        })

        # Print info
        print(f"\nTime Series: {series_id}")
        print(f"  → Chosen Model: {best_model}")
        print(f"  → {metric.upper()}: {best_score:.4f}")
        if naive_score is not None:
            print(f"  → Naive {metric.upper()}: {naive_score:.4f}")
        else:
            print(f"  → Naive {metric.upper()}: not available (model not evaluated)")

        all_forecasts.append(forecast_df)

    final_df = pd.concat(all_forecasts, axis=0)
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Forecasts saved to: {output_path}")

    return final_df

#--------------------------------------------------------

model_registry = {
    "Naive": Naive(),
    "RandomWalkWithDrift": RandomWalkWithDrift(),
    "AutoARIMA": AutoARIMA(),
    #"ETS": ETS(),
    "SeasonalNaive": SeasonalNaive(season_length=12),
    "HistoricAverage": HistoricAverage(),
    #"TSLM": TSLM() --should not be used
}

categories = {
    'Category 1': ["Naive", "RandomWalkWithDrift"],
    'Category 2': ["Naive", "SeasonalNaive"], #+ , "TSLM", "ETS"
    'Category 3': ["Naive", "AutoARIMA", "SeasonalNaive"], #+  "ETS", 
    'Category 4': ["Naive", "AutoARIMA", "RandomWalkWithDrift", "HistoricAverage"],
    'Category 5': ["Naive", "AutoARIMA", "HistoricAverage"] #+ , "TSLM"
}
#--------------------------------------------------------

file_name, product_classes_to_forecast, metric = get_user_inputs()
print(f"\nInputs received:\nFile: {file_name}\nProduct Classes to Forecast: {product_classes_to_forecast}\nMetric: {metric}")

ts_df = load_selected_timeseries(file_name, product_classes_to_forecast)
print(ts_df.head())

#Run cross validation
cv_results = run_cross_validation(ts_df, metric)
print(cv_results)

#Print accuracy metrics
if not cv_results.empty:
    accuracy_df = calculate_accuracy_metrics(cv_results, metric)
    print("Per-Series, Per-Model Accuracy:")
    print("Accuracy:\n")
    print(accuracy_df)
else:
    print("\nNo valid time series for cross-validation.")

# Optional: To get average accuracy across all series per model
#avg_accuracy = accuracy_df.mean(numeric_only=True).to_frame(name=metric.upper())
#avg_accuracy.index.name = 'model'
#avg_accuracy.reset_index(inplace=True)
#print("\nAverage accuracy across all series:")
#print(avg_accuracy)

#Select best models:
accuracy_df = accuracy_df.set_index("unique_id") # set unique_id as index
best_models_df = select_best_models(accuracy_df)

#Run forecast for best models:
forecast_df = forecast_best_models(ts_df, best_models_df, accuracy_df, horizon=12)



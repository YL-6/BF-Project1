import pandas as pd
import numpy as np
import utilsforecast.losses as ufl
import os # To create directory with plots
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    HistoricAverage,
    Naive,
    RandomWalkWithDrift,
    SeasonalNaive,
    SimpleExponentialSmoothingOptimized
)

#-------------------Input Data-----------------------------------

model_registry = {
    "AutoARIMA": AutoARIMA(), # Can handle seasonality, some types of trends and autoregression. If high seasonality, SARIMA should be used instead.
    "AutoETS": AutoETS(), # Handles seasonality and trends well. Not good with intermittent demand (e.g., lots of zeros), doesn't handle exogenous variables, and can struggle with highly non-linear series.
    "HistoricAverage": HistoricAverage(), # ideal for Time series with no trend, no seasonality. However highly sensitive to outliers.
    "SARIMA": AutoARIMA(season_length=12, alias="SARIMA"), # To be used instead of ARIMA if there is high seasonality.
    "Naive": Naive(), # Can be used for time series without seasonality and trend.
    "RandomWalkWithDrift": RandomWalkWithDrift(), # Ideal to model linear trends, but not seasonality. Sensitive to training window, i.e. chosen first and last observation.
    "SeasonalNaive": SeasonalNaive(season_length=12), # Applicable if high seasonality and no trend. Replicate last value in previous season, so highly sensitive to outliers.
    "SESOpt": SimpleExponentialSmoothingOptimized() # Useful for time series data without trend or seasonality — it smooths the series by weighting recent observations more heavily.
}

categories = {
    'Category 1': ["Naive", "RandomWalkWithDrift"], # Short time series (under 48 observations)
    'Category 2': ["Naive", "SeasonalNaive", "AutoETS"], # High seasonality and contains zeroes
    'Category 3': ["Naive", "SARIMA", "SeasonalNaive", "AutoETS"], # High seasonality and contains ony positive values
    'Category 4': ["Naive", "AutoARIMA", "RandomWalkWithDrift", "HistoricAverage", "SimpleExponentialSmoothing"], # Low or no seasonality and contains zeroes.
    'Category 5': ["Naive", "AutoARIMA", "HistoricAverage", "SESOpt", "AutoETS"] # All remaining
}

#---------------------Functions-----------------------------------

def get_user_inputs():
    # Get file name
    file_name = input("Enter the file name (with extension, e.g., data.csv): ").strip()

    # Get list of time series IDs to forecast
    product_classes_input = input("Enter time series IDs to forecast, separated by commas: ").strip()
    # Split input into list of product class IDs
    product_classes_to_forecast = [id_.strip() for id_ in product_classes_input.split(",") if id_.strip()]
    # Validate number of product classes
    if len(product_classes_to_forecast) == 0 or len(product_classes_to_forecast) > 5:
        raise ValueError("Incorrect number of product classes. Input 1 up to 5 product classes.")

    # Get metric and validate
    metric = input("Enter the metric ('mape' or 'mse'): ").strip().lower()
    if metric not in ['mape', 'mse']:
        raise ValueError("Invalid metric. Must be 'mape' or 'mse'.")

    return file_name, product_classes_to_forecast, metric


def load_selected_timeseries(file_name: str, product_classes: list[str]) -> pd.DataFrame:
    df = pd.read_csv(file_name)

    # Drop rows with missing critical columns
    df.dropna(subset=["product_class", "Month", "sales_volume"], inplace=True)

    # Convert Month to datetime; drop rows with invalid dates
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df.dropna(subset=["Month"], inplace=True)

    # Filter for requested product classes
    df = df[df["product_class"].isin(product_classes)]

    # Convert sales_volume to numeric; drop rows where conversion failed
    df["sales_volume"] = pd.to_numeric(df["sales_volume"], errors="coerce")
    df.dropna(subset=["sales_volume"], inplace=True)

    # Sort data by Month (date)
    df = df.sort_values("Month")

    # Rename columns and select the needed ones
    formatted_df = df.rename(columns={
        "product_class": "unique_id",
        "Month": "ds",
        "sales_volume": "y"
    })[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)

    return formatted_df


def quantify_seasonality_strength(ts_df: pd.DataFrame, freq: int = 12) -> pd.Series:
    """
    Quantifies the strength of seasonality and residual variability in a time series.

    Returns:
        pd.Series with 'strength_seasonality' and 'residual_variability'
    """
    ts = ts_df.set_index("ds")["y"]
    ts = ts.asfreq("MS")  # Ensure frequency is monthly

    try:
        stl = STL(ts, period=freq, robust=True)
        result = stl.fit()
        fs = max(0, 1 - np.var(result.resid) / np.var(result.resid + result.seasonal))
        rv = np.std(result.resid / ts.mean())
        return pd.Series({
            "strength_seasonality": round(fs, 2),
            "residual_variability": round(rv, 2)
        })
    except Exception as e:
        return pd.Series({
            "strength_seasonality": 0.0,
            "residual_variability": np.nan
        })


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
    
    result = quantify_seasonality_strength(ts_df, freq=12)
    strength = result["strength_seasonality"]

    has_zeros = (y == 0).any()
    strong_seasonality = strength >= 0.64
    low_seasonality = strength < 0.4

    if strong_seasonality and has_zeros:
        return "Category 2"
    elif strong_seasonality and (y > 0).all():
        return "Category 3"
    elif low_seasonality and has_zeros:
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
    metric: str,
    horizon: int = 12,
    output_path: str = "final_forecasts.csv"
) -> pd.DataFrame:
    """
    Forecasts using the best model and a naive benchmark for each time series.
    Saves forecasts to CSV and returns the DataFrame with standardized format.
    """
    all_forecasts = []

    # Create a lookup dict for best model per unique_id
    best_models_lookup = best_models_df.set_index('unique_id')['best_model'].to_dict()

    for unique_id, group_df in ts_df.groupby("unique_id"):
        group_df = group_df.sort_values("ds")

        # Forecast using the best model
        model_name = best_models_lookup[unique_id]
        model = model_registry.get(model_name)

        print(f"\nForecasting for {unique_id} using model: {model_name}")
        sf = StatsForecast(models=[model], freq='MS')
        best_forecast = sf.forecast(df=group_df, h=horizon)
        print(best_forecast.head())

        # Rename forecast column to 'y_hat'
        best_col = [col for col in best_forecast.columns if col not in ['ds', 'unique_id']][0]
        best_forecast = best_forecast.rename(columns={best_col: 'y_hat'})
        best_forecast['unique_id'] = unique_id
        best_forecast['model'] = model_name
        all_forecasts.append(best_forecast)

        # Forecast using Naive model
        if model_name != "Naive":
            sf_naive = StatsForecast(models=[Naive()], freq='MS')
            naive_forecast = sf_naive.forecast(df=group_df, h=horizon)

            naive_col = [col for col in naive_forecast.columns if col not in ['ds', 'unique_id']][0]
            naive_forecast = naive_forecast.rename(columns={naive_col: 'y_hat'})
            naive_forecast['unique_id'] = unique_id
            naive_forecast['model'] = 'Naive'
            all_forecasts.append(naive_forecast)

    if all_forecasts:
        final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
        final_forecast_df.to_csv(output_path, index=False)
        print("final forecast ok.")
        return final_forecast_df
    else:
        print("No forecasts generated.")
        return pd.DataFrame()


def plot_forecasts(ts_df, forecast_df, best_models_df, metric, plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)

    for _, row in best_models_df.iterrows():
        series_id = row["unique_id"]
        best_model = row["best_model"]
        best_score = row.get("best_metric", None)

        # Historical data
        history = ts_df[ts_df["unique_id"] == series_id]

        # Forecasts for this series and best model
        model_forecast = forecast_df[
            (forecast_df["unique_id"] == series_id) & 
            (forecast_df["model"] == best_model)
        ]

        # Forecasts for naive model, only if it's NOT the best model
        if best_model != "Naive":
            naive_forecast = forecast_df[
                (forecast_df["unique_id"] == series_id) & 
                (forecast_df["model"] == "Naive")
            ]
        else:
            naive_forecast = pd.DataFrame()  # empty, so it won't be plotted again

        if model_forecast.empty:
            print(f"⚠️ No forecast available for {series_id} with model {best_model}")
            continue

        plt.figure(figsize=(10, 5))
        plt.plot(history["ds"], history["y"], label="Actuals", marker='o')
        plt.plot(model_forecast["ds"], model_forecast["y_hat"], 
                 label=f"{best_model} Forecast", linestyle="--", marker='x', color="blue")

        # Plot naive forecast if it's not already the best
        if not naive_forecast.empty:
            plt.plot(naive_forecast["ds"], naive_forecast["y_hat"], 
                     label="Naive Forecast", linestyle=":", marker='s', color="orange")

        title = f"{series_id} Forecast ({best_model})"
        if best_score is not None:
            title += f"\n{metric.upper()}: {best_score:.2f}"

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid(True)

        filename = os.path.join(plot_dir, f"forecast_{series_id}_{best_model}.png")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        print(f"📊 Saved forecast plot to {filename}")


def create_summary_table(ts_df, best_models_df, accuracy_df):
    # Get category per series
    category_assignments = ts_df.groupby("unique_id").apply(select_category).rename("category")

    # Get number of observations per series
    counts = ts_df.groupby("unique_id").size().rename("num_observations")

    # Extract naive accuracy from accuracy_df
    # accuracy_df has index unique_id and columns = model names
    naive_acc = accuracy_df['Naive'].rename("naive_metric")

    # Merge all info
    summary_df = best_models_df.set_index('unique_id').join([category_assignments, counts, naive_acc])

    # Rename columns for clarity
    summary_df = summary_df.rename(columns={
        'best_model': 'best_model_chosen',
        'best_metric': 'best_model_metric_value'
    })

    # Reset index to have unique_id as a column if desired
    summary_df = summary_df.reset_index()

    return summary_df


#------------------Main script--------------------------------------

if __name__ == "__main__":
    # Step 1: Get user input
    file_name, product_classes_to_forecast, metric = get_user_inputs()
    print(f"\nInputs received:\nFile: {file_name}\nProduct Classes to Forecast: {product_classes_to_forecast}\nMetric: {metric}")

    # Step 2: Load time series
    ts_df = load_selected_timeseries(file_name, product_classes_to_forecast)
    if ts_df.empty:
        print("❌ No time series found for the selected product classes.")
        exit(1)
    print("✅ Loaded time series:\n", ts_df.head(), "\n")

    #print(ts_df.groupby('unique_id').get_group('C3029'))

    # Step 3: Minimum series length
    ts_min_length = ts_df.groupby("unique_id").size().min()
    if ts_min_length < 48: # Get length of shortest time series
        print("Shortest time series length:", ts_min_length)
    else:
        ts_min_length = 48  # force to minimum if needed

    # Step 4: Cross-validation
    cv_results = run_cross_validation(ts_df, metric, ts_min_length)
    if cv_results.empty:
        print("⚠️ No valid time series for cross-validation.")
        exit(1)

    # Step 5: Accuracy calculation
    accuracy_df = calculate_accuracy_metrics(cv_results, metric)
    accuracy_df = accuracy_df.set_index("unique_id")
    print("📈 Per-Series, Per-Model Accuracy:\n", accuracy_df)

    # Step 6: Best model selection
    best_models_df = select_best_models(accuracy_df)
    print("🏆 Best models selected:\n", best_models_df)

    # Step 7: Forecasts (best models + Naive)
    forecast_df = forecast_best_models(
        ts_df, best_models_df, metric=metric, horizon=12
    )

    # Step 8: Plot forecasts
    plot_forecasts(ts_df, forecast_df, best_models_df, metric)

    #Step 10: export tables
    summary_df = create_summary_table(ts_df, best_models_df, accuracy_df)
    print("\nSummary Table:")
    print(summary_df)

    with pd.ExcelWriter('output.xlsx') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        cv_results.to_excel(writer, sheet_name='Cross_Validation', index=False)
        accuracy_df.to_excel(writer, sheet_name='Accuracy', index=False)
        best_models_df.to_excel(writer, sheet_name='Best_Models', index=False)
        forecast_df.to_excel(writer, sheet_name='Forecasts', index=False)


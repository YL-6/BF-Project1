import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

def get_user_inputs():
    # Get file name
    file_name = input("Enter the file name (with extension, e.g., data.csv): ").strip()

    # Get list of time series IDs to forecast
    product_classes_input = input("Enter time series IDs to forecast, separated by commas: ").strip()
    product_classes_to_forecast = [id_.strip() for id_ in product_classes_input.split(",") if id_.strip()]

    # Get metric and validate
    metric = input("Enter the metric ('mape' or 'mse'): ").strip().lower()
    if metric not in ['mape', 'mse']:
        raise ValueError("Invalid metric. Must be 'mape' or 'mse'.")

    return file_name, product_classes_to_forecast, metric

def load_selected_timeseries(file_name: str, product_classes: list[str]) -> dict:
    """
    Load sales data from a CSV and return time series only for specified product classes.

    Args:
        file_name (str): Path to the CSV file.
        product_classes (list[str]): List of product_class names to extract.

    Returns:
        dict: Dictionary of {product_class: time series (pd.Series)}.
    """
    # Load the CSV
    df = pd.read_csv(file_name)

    # Drop rows with missing essential values
    df.dropna(subset=["product_class", "Month", "sales_volume"], inplace=True)

    # Convert Month to datetime
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df.dropna(subset=["Month"], inplace=True)

    # Filter for selected product_classes
    df = df[df["product_class"].isin(product_classes)]

    # Sort by date
    df = df.sort_values("Month")

    # Group and extract time series with product class as name
    time_series_dict = {
    product_class: group.set_index("Month")["sales_volume"].sort_index().rename(product_class)
    for product_class, group in df.groupby("product_class")
    }

    return time_series_dict

def check_seasonality_strength(ts: pd.Series, freq: int = 12, threshold: float = 0.5) -> str:
    """
    Determines if a time series has high or low seasonality.

    Args:
        ts (pd.Series): Time series with a datetime index.
        freq (int): Number of periods in a season (e.g., 12 for monthly data with yearly seasonality).
        threshold (float): Threshold for determining "high" vs "low" seasonality (0 to 1 scale).

    Returns:
        str: "high" or "low" seasonality.
    """
    # Check for enough data
    if len(ts) < 2 * freq:
        return "low"  # Not enough data to assess seasonality

    # Decompose the time series
    decomposition = seasonal_decompose(ts, model='additive', period=freq, extrapolate_trend='freq')

    # Compute strength of seasonality
    resid = decomposition.resid.dropna()
    seasonal = decomposition.seasonal.loc[resid.index]

    var_resid = np.var(resid)
    var_combined = np.var(resid + seasonal)

    if var_combined == 0:
        return "low"  # Avoid division by zero or flat series

    strength = 1 - (var_resid / var_combined)

    return "high" if strength >= threshold else "low"

def select_category(ts: pd.Series):
    """Determines to which category a time series belongs."""
    if len(ts) < 48:
        return "Category 1"
    elif check_seasonality_strength(ts) == "high" and (ts == 0).any():
        return "Category 2"
    elif check_seasonality_strength(ts) == "high" and (ts > 0).all():
        return "Category 3"
    elif check_seasonality_strength(ts) == "low" and (ts == 0).any():
        return "Category 4"
    else:
        return "Category 5"

#--------------------------------------------------------

categories = {
    'Category 1': ["naive", "drift"],
    'Category 2': ["naive", "ETS", "seasonal naive", "TSLM"],
    'Category 3': ["naive", "ARIMA", "ETS", "seasonal naive"],
    'Category 4': ["naive", "ARIMA", "drift", "mean"],
    'Category 5': ["naive", "ARIMA", "mean", "TSLM"]
}


file_name, product_classes_to_forecast, metric = get_user_inputs()
print(f"\nInputs received:\nFile: {file_name}\nProduct Classes to Forecast: {product_classes_to_forecast}\nMetric: {metric}")

ts_dict = load_selected_timeseries(file_name, product_classes_to_forecast)

for p_class, series in ts_dict.items():
    print(f"\nTime Series for {p_class}:")
    print(series)
    print(select_category(series))





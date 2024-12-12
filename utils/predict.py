import numpy as np
import datetime
import pandas as pd
from pmdarima import auto_arima

def datetime_to_number(date: str):
    """Convert a date string 'YYYY-MM-DD' to a relative day number."""
    date_number = datetime.datetime.strptime(date, "%Y-%m-%d")
    base_number = datetime.datetime.strptime("2024-1-1", "%Y-%m-%d")
    return (date_number - base_number).days

def predict_future_values(data, forecast_days=5):
    """
    Use auto_arima from pmdarima to fit a suitable ARIMA/SARIMA model for the time series,
    then predict future values for the specified number of days.

    Parameters:
    data: dict, keys are date strings 'YYYY-MM-DD', values are integer counts
    forecast_days: int, number of days to predict into the future

    Returns:
    predictions: dict, keys are future date strings 'YYYY-MM-DD', values are predicted integers (≥0)
    """
    if not data:
        return {}

    # Sort data by date
    sorted_dates = sorted(data.keys(), key=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d"))
    start_date = sorted_dates[0]
    end_date = sorted_dates[-1]
    
    # Create a full date range to ensure continuity in the time series
    full_range = pd.date_range(start=start_date, end=end_date, freq='D')
    ts = pd.Series(0, index=full_range, dtype=float)
    for d in data:
        ts[pd.to_datetime(d)] = data[d]

    # Simple smoothing: optional step to reduce noise (moving average over 3 days)
    # This is a mild smoothing to handle noisy data. You can comment this out if not needed.
    ts_smoothed = ts.rolling(window=3, min_periods=1).mean()
    
    # Fit the time series with auto_arima to find the best parameters
    model = auto_arima(ts_smoothed, 
                       start_p=1, start_q=1, 
                       max_p=5, max_q=5, 
                       seasonal=False,
                       trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
    
    # Predict the future values
    forecast = model.predict(n_periods=forecast_days)
    # Construct future dates
    last_date = pd.to_datetime(end_date)
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, forecast_days+1)]
    
    # Convert forecast results to dict with non-negative integers
    predictions = {}
    for d, v in zip(future_dates, forecast):
        predictions[d.strftime("%Y-%m-%d")] = max(int(round(v)), 0)

    return predictions

if __name__ == '__main__':
    data = {
        '2024-06-15': 1, '2024-06-18': 1, '2024-06-22': 1,
        '2024-06-23': 1, '2024-07-01': 3, '2024-07-02': 4,
        '2024-07-03': 4, '2024-07-04': 14
    }
    preds = predict_future_values(data)
    print(preds)

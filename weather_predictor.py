import json
import pandas as pd
import numpy as np
import warnings
import os
import glob
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

warnings.filterwarnings("ignore", message="No supported index is available.")
warnings.filterwarnings("ignore", message="A date index has been provided, but it has no associated frequency information")
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Known canonical aliases for Indian cities
CITY_ALIASES = {
    "bangalore": "bengaluru",
    "calcutta": "kolkata",
    "bombay": "mumbai",
    "madras": "chennai",
    "gurgaon": "gurugram",
    "banaras": "varanasi",
    "kashi": "varanasi",
    "new delhi": "delhi",
    "dilli": "delhi",
    "mysore": "bengaluru",
    "ncr": "delhi",
}

# In-memory cache for fitted SARIMAX models to speed up repeat predictions
_MODEL_CACHE = {}

def normalize_city_name(city_name: str) -> str:
    """Normalize input city string into canonical name."""
    if not city_name:
        return ""
    clean = city_name.strip().lower()
    clean = clean.replace("-", " ").replace("_", " ")
    return CITY_ALIASES.get(clean, clean).replace(" ", "_")

def get_available_cities():
    """Returns a list of all cities available in the data/ folder."""
    if not os.path.exists(DATA_DIR):
        return []
    cities = []
    for f in glob.glob(os.path.join(DATA_DIR, "*.json")):
        basename = os.path.basename(f)
        if basename == "nasa_power_1985_2025.json":
            continue
        city_name = os.path.splitext(basename)[0].replace("_", " ").title()
        cities.append(city_name)
    return sorted(cities)

def resolve_data_path(file_path: str) -> str:
    """
    Resolves data file path by checking:
    1. Direct path as provided
    2. Normalized city filename in data/ directory
    3. data/ directory relative to current working directory
    """
    if os.path.exists(file_path):
        return file_path

    base_name = os.path.basename(file_path)
    city_key = os.path.splitext(base_name)[0]
    normalized_key = normalize_city_name(city_key)

    candidates = [
        os.path.join(DATA_DIR, f"{normalized_key}.json"),
        os.path.join(DATA_DIR, base_name),
        os.path.join("data", f"{normalized_key}.json"),
        os.path.join("data", base_name),
    ]

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    return file_path

def load_and_preprocess_data(file_path: str):
    resolved = resolve_data_path(file_path)
    if not os.path.exists(resolved):
        available = ", ".join(get_available_cities())
        raise FileNotFoundError(
            f"City dataset '{os.path.basename(file_path)}' not found. "
            f"Available cities in data catalog: {available}"
        )

    with open(resolved, 'r', encoding='utf-8') as f:
        data = json.load(f)

    properties = data.get('properties', {}).get('parameter', {})
    available_params = list(properties.keys()) if properties else []

    if 'T2M' not in available_params:
        raise KeyError(f"Expected 'T2M' (temperature) parameter in JSON. Found: {available_params}")

    t2m_data = properties['T2M']

    # Priority order for precipitation parameters from NASA POWER
    precip_keys = ['PRECTOTCORR', 'PRECTOT', 'TP', 'PRECT', 'RAIN', 'PRCP', 'TOTAL_PRECIPITATION']
    precip_data = None
    precip_key_used = None
    for key in precip_keys:
        if key in available_params:
            precip_data = properties[key]
            precip_key_used = key
            break

    has_precip = bool(precip_data)

    # DataFrame for T2M
    df_t2m = pd.DataFrame(list(t2m_data.items()), columns=['Date', 'T2M'])
    df_t2m['Date'] = pd.to_datetime(df_t2m['Date'], format='%Y%m%d', errors='coerce')
    df_t2m = df_t2m.dropna(subset=['Date']).set_index('Date')
    df_t2m = df_t2m.asfreq('D')
    df_t2m['T2M'] = df_t2m['T2M'].ffill().bfill()

    # DataFrame for Precipitation
    df_tp = None
    if has_precip:
        df_tp = pd.DataFrame(list(precip_data.items()), columns=['Date', precip_key_used])
        df_tp['Date'] = pd.to_datetime(df_tp['Date'], format='%Y%m%d', errors='coerce')
        df_tp = df_tp.dropna(subset=['Date']).set_index('Date')
        df_tp = df_tp.asfreq('D')
        df_tp[precip_key_used] = df_tp[precip_key_used].fillna(0).clip(lower=0)

    return df_t2m, df_tp, has_precip, precip_key_used

def fallback_seasonal_forecast(series: pd.Series, future_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Robust seasonal fallback forecaster using day-of-year historical medians
    combined with a 30-day baseline trend. Used when SARIMAX numerical fitting fails.
    """
    if series is None or len(series) == 0:
        return pd.Series(25.0, index=future_dates)

    # Group by (month, day) for climate normal baseline
    df = pd.DataFrame({'value': series})
    df['month'] = df.index.month
    df['day'] = df.index.day
    seasonal_median = df.groupby(['month', 'day'])['value'].median()

    # Calculate recent 30-day drift
    recent_mean = series.iloc[-30:].mean() if len(series) >= 30 else series.mean()
    overall_mean = series.mean()
    drift = (recent_mean - overall_mean) * 0.3  # dampened anomaly persistence

    forecast_values = []
    for dt in future_dates:
        m, d = dt.month, dt.day
        val = seasonal_median.get((m, d), overall_mean) + drift
        forecast_values.append(val)

    return pd.Series(forecast_values, index=future_dates)

def train_model(data: pd.Series, order=(1,1,1), seasonal_order=(1,1,1,7), window_size=1825, cache_key=None):
    """
    Fits SARIMAX on a focused window (default last 5 years = 1825 days).
    This dramatically accelerates training from 30s to ~1s and prevents convergence failures.
    """
    if data is None or len(data) == 0 or not HAS_STATSMODELS:
        return None

    if cache_key and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Focus on the most recent 5 years of daily measurements for speed and stability
    train_slice = data.iloc[-window_size:] if len(data) > window_size else data

    if train_slice.index.freq is None:
        train_slice = train_slice.asfreq('D')

    try:
        model = SARIMAX(
            train_slice,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False, maxiter=50)
        if cache_key:
            _MODEL_CACHE[cache_key] = model_fit
        return model_fit
    except Exception as e:
        print(f"[Pipeline Warning] SARIMAX training optimization note: {e}. Fallback forecaster will ensure stability.")
        return None

def evaluate_model(model_fit, test_data):
    if model_fit is None or test_data is None:
        return None, None, None, None
    try:
        start = len(model_fit.fittedvalues)
        end = start + len(test_data) - 1
        predictions = model_fit.predict(start=start, end=end, dynamic=False)
        predictions.index = test_data.index
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mape = np.mean(np.abs((test_data - predictions) / np.where(test_data == 0, 1e-5, test_data))) * 100
        return predictions, mae, rmse, mape
    except Exception:
        return None, None, None, None

def forecast_for_date(model_fit_temp, model_fit_precip, last_date, target_date, has_precip, precip_key_used, df_t2m=None, df_tp=None):
    """
    Dual-mode forecaster:
    1. Historical lookup: If target_date <= last_date, returns actual historical measurements.
    2. Predictive forecast: If target_date > last_date, runs SARIMAX with robust seasonal fallback.
    """
    # 1. Historical Lookup Mode
    if target_date <= last_date:
        if df_t2m is not None and target_date in df_t2m.index:
            context_start = max(target_date - pd.Timedelta(days=3), df_t2m.index[0])
            context_temp = df_t2m.loc[context_start:target_date, 'T2M']
            context_precip = None
            if has_precip and df_tp is not None and target_date in df_tp.index:
                context_precip = df_tp.loc[context_start:target_date, precip_key_used]
            return context_temp, context_precip, "historical_actual"

    # 2. Future Forecast Mode
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=target_date, freq='D')
    n_steps = len(future_dates)
    method_used = "sarimax_model"

    # Temperature forecast
    if model_fit_temp is not None:
        try:
            forecast_temp = model_fit_temp.predict(
                start=len(model_fit_temp.fittedvalues),
                end=len(model_fit_temp.fittedvalues) + n_steps - 1
            )
            forecast_temp.index = future_dates
        except Exception:
            forecast_temp = fallback_seasonal_forecast(df_t2m['T2M'], future_dates)
            method_used = "seasonal_fallback"
    else:
        forecast_temp = fallback_seasonal_forecast(df_t2m['T2M'] if df_t2m is not None else None, future_dates)
        method_used = "seasonal_fallback"

    # Precipitation forecast
    forecast_precip = None
    if has_precip:
        if model_fit_precip is not None:
            try:
                forecast_precip = model_fit_precip.predict(
                    start=len(model_fit_precip.fittedvalues),
                    end=len(model_fit_precip.fittedvalues) + n_steps - 1
                )
                forecast_precip.index = future_dates
                forecast_precip = forecast_precip.clip(lower=0)
            except Exception:
                forecast_precip = fallback_seasonal_forecast(df_tp[precip_key_used], future_dates).clip(lower=0)
        else:
            forecast_precip = fallback_seasonal_forecast(df_tp[precip_key_used] if df_tp is not None else None, future_dates).clip(lower=0)

    context_start = max(target_date - pd.Timedelta(days=3), future_dates[0])
    context_temp = forecast_temp[context_start:target_date]
    context_precip = forecast_precip[context_start:target_date] if has_precip and forecast_precip is not None else None

    return context_temp, context_precip, method_used

def will_it_rain(precip_forecast, temp_forecast, date, has_precip):
    """Calibrated rain prediction and details."""
    month = date.month
    is_monsoon = 6 <= month <= 9

    if has_precip and precip_forecast is not None:
        precip_val = float(precip_forecast)
        will_rain = precip_val >= 0.5
        if precip_val >= 15.0:
            status = "Heavy Rain"
        elif precip_val >= 5.0:
            status = "Moderate Rain"
        elif precip_val >= 0.5:
            status = "Light Rain"
        else:
            status = "Dry / Clear"
        return will_rain, f"{status} (Precip: {precip_val:.2f} mm)"

    temp = temp_forecast if temp_forecast is not None else 25
    likely_rain = is_monsoon and 20 <= temp <= 32
    rain_status = "Likely (Monsoon season)" if likely_rain else "Unlikely"
    return likely_rain, f"Rain: {rain_status} (heuristic estimate)"

def main():
    print(f"Available Cities: {', '.join(get_available_cities())}")
    city_name = input("Enter city name (e.g., Bangalore, Mumbai): ").strip()
    target_date_str = input("Enter target date (YYYY-MM-DD): ").strip()

    try:
        target_date = pd.to_datetime(target_date_str)
    except Exception:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    normalized_city = normalize_city_name(city_name)
    file_name = f"{normalized_city}.json"

    try:
        df_t2m, df_tp, has_precip, precip_key_used = load_and_preprocess_data(file_name)
    except Exception as e:
        print(f"Error loading city data: {e}")
        return

    time_series_temp = df_t2m['T2M']
    time_series_precip = df_tp[precip_key_used] if has_precip else None
    last_date = time_series_temp.index[-1]

    model_fit_temp = train_model(time_series_temp, cache_key=f"{normalized_city}_T2M")
    model_fit_precip = train_model(time_series_precip, cache_key=f"{normalized_city}_PRECIP") if has_precip else None

    forecast_context_temp, forecast_context_precip, method = forecast_for_date(
        model_fit_temp, model_fit_precip, last_date, target_date,
        has_precip, precip_key_used, df_t2m=df_t2m, df_tp=df_tp
    )

    print(f"\n--- Weather Details for {city_name.title()} on {target_date.strftime('%Y-%m-%d')} ---")
    print(f"Pipeline Mode: {method}")

    for date in forecast_context_temp.index:
        temp = forecast_context_temp[date]
        day_label = "Target Date" if date == target_date else f"Context ({(target_date - date).days} days prior)"
        precip_val = forecast_context_precip[date] if has_precip and forecast_context_precip is not None else 0.0
        will_rain, rain_details = will_it_rain(precip_val, temp, date, has_precip)
        print(f"{date.strftime('%Y-%m-%d')} ({day_label}): {temp:.2f} °C | {rain_details}")

if __name__ == "__main__":
    main()

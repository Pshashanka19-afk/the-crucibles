"""
NASA POWER API Weather Data Fetcher
Retrieves daily temperature (T2M), wind speed (WS2M), and precipitation (PRECTOT)
data for a given latitude and longitude from NASA POWER API and saves to JSON in data/
"""

import os
import json
import requests
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))

def fetch_weather_data(lat=12.97, lon=77.59, start="19850101", end="20251001", output_filename="nasa_power_1985_2025.json"):
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, output_filename)

    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=T2M,WS2M,PRECTOT&community=RE&longitude={lon}&latitude={lat}&"
        f"start={start}&end={end}&format=JSON"
    )

    print(f"Fetching data from NASA POWER API for coords ({lat}, {lon})...")
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Data successfully saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical weather data from NASA POWER API")
    parser.add_argument("--lat", type=float, default=12.97, help="Latitude (default: 12.97 - Bangalore)")
    parser.add_argument("--lon", type=float, default=77.59, help="Longitude (default: 77.59 - Bangalore)")
    parser.add_argument("--start", type=str, default="19850101", help="Start date YYYYMMDD (default: 19850101)")
    parser.add_argument("--end", type=str, default="20251001", help="End date YYYYMMDD (default: 20251001)")
    parser.add_argument("--out", type=str, default="nasa_power_1985_2025.json", help="Output filename inside data/")

    args = parser.parse_args()
    fetch_weather_data(lat=args.lat, lon=args.lon, start=args.start, end=args.end, output_filename=args.out)

# Weather Foreseen 🌦️

**Team The Crucibles | NASA Space Apps Hackathon**

Weather Foreseen is an end-to-end meteorological prediction and natural language query platform. It combines historical satellite and surface weather datasets from the **NASA POWER API** with **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors)** time-series forecasting models and an integrated **Google Gemini AI** weather assistant.

---

## 📁 Project Structure

```text
the-crucibles/
├── data/                           # Historical NASA POWER datasets (JSON format)
│   ├── ahmedabad.json
│   ├── amritsar.json
│   ├── bengaluru.json
│   ├── chennai.json
│   ├── darjeeling.json
│   ├── delhi.json
│   ├── gurugram.json
│   ├── jaipur.json
│   ├── kochi.json
│   ├── kolkata.json
│   ├── mumbai.json
│   ├── ooty.json
│   ├── patna.json
│   ├── srinagar.json
│   ├── varanasi.json
│   └── nasa_power_1985_2025.json
│
├── templates/                      # Flask HTML templates
│   └── index.html                  # Main responsive web dashboard UI
│
├── static/                         # Static assets (icons, styles, images)
│   └── thecrucibles.svg            # Weather Foreseen vector icon/favicon
│
├── scripts/                        # Data acquisition & utility scripts
│   └── fetch_nasa_data.py          # NASA POWER API daily point data fetcher
│
├── archive/                        # Preserved earlier prototypes & iterations
│   ├── final.html                  # Pre-export dropdown HTML version
│   └── try.py                      # Prototype chatbot backend
│
├── app.py                          # Flask application server & API endpoints
├── weather_predictor.py            # SARIMAX forecasting engine & preprocessor
├── requirements.txt                # Python project dependencies
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Git ignore rules for Python, cache & env files
└── README.md                       # Project documentation
```

---

## 🚀 Features

- **SARIMAX Time-Series Forecasting**: Machine learning models trained on decades of NASA meteorological measurements to predict daily temperature (`T2M`) and precipitation (`PRECTOT`).
- **Conversational Weather Assistant**: Natural language weather queries powered by Google Gemini (e.g., *"Will it rain in Delhi next weekend?"* or *"What is the expected temperature in Bangalore on Christmas?"*).
- **Interactive Web Dashboard**:
  - Live weather forecast view with responsive animations and themes (Light/Dark mode).
  - City selection with pre-loaded datasets across 15+ Indian cities.
  - Multi-format prediction export: **JSON**, **CSV**, and **PDF** (using jsPDF).
- **Automated Data Fetching**: Command-line script to retrieve new location datasets directly from NASA POWER API.

---

## 🛠️ Getting Started

### 1. Prerequisites

- Python 3.9+ (Python 3.10 - 3.13 recommended)
- `pip` or `uv` package manager

### 2. Clone and Setup Environment

```bash
# Navigate to project repository
cd the-crucibles

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file and add your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env` with your values:
```env
GENAI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your_flask_secret_key_here
```

*(If no `.env` is provided, a default fallback key is configured for demo purposes.)*

### 4. Run the Application

Start the Flask server:

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 📡 API Endpoints

### `GET /`
Serves the interactive frontend web dashboard.

### `POST /predict`
Generates a weather forecast for a selected city and future date.

- **Request Body (JSON)**:
  ```json
  {
    "city": "Bengaluru",
    "date": "2026-12-31"
  }
  ```
- **Response (JSON)**:
  ```json
  {
    "temperature": 24.35,
    "rainfall": 1.20,
    "humidity": 65,
    "rain_chance": "40%",
    "rain_details": "Rain: Yes (Precip: 1.20 mm)",
    "will_rain": "Yes"
  }
  ```

### `POST /chat`
Conversational chat assistant that extracts intent, dates, and locations from text queries and returns intelligent forecast summaries.

- **Request Body (JSON)**:
  ```json
  {
    "message": "What will the temperature be in Mumbai on 2026-11-20?"
  }
  ```
- **Response (JSON)**:
  ```json
  {
    "reply": "Forecast for Mumbai on 2026-11-20: Temperature 29.5°C, Rain: 0.0 mm. It will feel warm. It should remain dry."
  }
  ```

---

## 📊 Fetching New NASA Datasets

To download fresh weather datasets from NASA for any coordinate:

```bash
# Run with custom coordinates (e.g., Lat: 28.61, Lon: 77.20 for Delhi):
python scripts/fetch_nasa_data.py --lat 28.61 --lon 77.20 --start 19850101 --end 20251001 --out new_city.json
```

---

## 👥 Authors & Acknowledgments

- **Team The Crucibles**
- NASA Space Apps Challenge
- Data provided by [NASA POWER Project](https://power.larc.nasa.gov/)

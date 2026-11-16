from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os, json, re
import google.generativeai as genai
from weather_predictor import (
    load_and_preprocess_data,
    train_model,
    forecast_for_date,
    will_it_rain,
    normalize_city_name,
    get_available_cities,
)
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = os.getenv("SECRET_KEY", "weather-forseen-secret-key")

CORS(app)

# Load environment variables if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

api_key = os.getenv("GENAI_API_KEY", "AIzaSyCQx08C1YOYiw7Q8AaBxChwmIng02LSi4Q")
if api_key:
    genai.configure(api_key=api_key)

try:
    chat_model = genai.GenerativeModel("gemini-2.5-flash")
except Exception:
    try:
        chat_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        chat_model = None

@app.route("/")
def index():
    session.clear()
    return render_template("index.html")

@app.route("/cities", methods=["GET"])
def list_cities():
    """Returns available pre-loaded cities for UI autocomplete and search."""
    return jsonify({
        "status": "success",
        "cities": get_available_cities()
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Pipeline health and diagnostics endpoint."""
    return jsonify({
        "status": "healthy",
        "project": "Weather Foreseen",
        "event": "NASA International Space Apps Challenge 2025",
        "team": "The Crucibles (1st Hackathon Milestone)",
        "available_cities_count": len(get_available_cities()),
        "gemini_ai_configured": chat_model is not None
    })

@app.route("/chat", methods=["POST"])
def chat():
    if "conversation_history" not in session:
        session["conversation_history"] = ""
    if "current_location" not in session:
        session["current_location"] = None
    if "current_date" not in session:
        session["current_date"] = None
    if "last_forecast" not in session:
        session["last_forecast"] = None

    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "Please type something!"})

    session["conversation_history"] += f"User: {user_input}\n"
    user_lower = user_input.lower()
    extracted_location = session.get("current_location")
    extracted_date = session.get("current_date")

    # Extract location and date using LLM if available
    if chat_model:
        try:
            extraction_prompt = f"""
            Extract location and date from this user query as JSON.
            Return JSON exactly in the form: {{ "location": "<city or null>", "date": "<YYYY-MM-DD or null>" }}
            User query: \"{user_input}\"
            """
            extraction_response = chat_model.generate_content(extraction_prompt)
            match = re.search(r"\{[\s\S]*\}", extraction_response.text.strip())
            if match:
                parsed = json.loads(match.group(0))
                loc = parsed.get("location")
                if loc and isinstance(loc, str) and loc.strip().lower() != "null":
                    extracted_location = loc.strip()
                dt = parsed.get("date")
                if dt and isinstance(dt, str) and dt.strip().lower() != "null":
                    iso = pd.to_datetime(dt, dayfirst=False, errors="coerce")
                    if not pd.isna(iso):
                        extracted_date = iso.strftime("%Y-%m-%d")
            session["current_location"] = extracted_location
            session["current_date"] = extracted_date
        except Exception:
            pass

    # Quick contextual memory answers
    if "date" in user_lower and any(w in user_lower for w in ("ask", "asked", "just", "which", "what", "did", "previous", "last")):
        reply = f"You asked for {session['current_date']}." if session.get("current_date") else "I don't have a recorded date yet."
        session["conversation_history"] += f"Assistant: {reply}\n"
        return jsonify({"reply": reply})

    if any(w in user_lower for w in ("which city", "which location", "what city", "what location", "which place")) or \
       ("location" in user_lower and any(w in user_lower for w in ("ask", "asked", "did", "which", "what", "last"))):
        reply = f"You asked about {session['current_location']}." if session.get("current_location") else "I don't have a recorded location yet."
        session["conversation_history"] += f"Assistant: {reply}\n"
        return jsonify({"reply": reply})

    if any(w in user_lower for w in ("repeat", "remind", "again", "show me")) and any(w in user_lower for w in ("forecast", "weather", "prediction", "result", "last")):
        last = session.get("last_forecast")
        reply = last["text"] if last and isinstance(last, dict) and last.get("text") else "I don't have a saved forecast to repeat."
        session["conversation_history"] += f"Assistant: {reply}\n"
        return jsonify({"reply": reply})

    if any(w in user_lower for w in ("hackathon", "nasa", "crucible", "team", "who made")):
        reply = (
            "Weather Foreseen was built by team 'The Crucibles' for the NASA International Space Apps Challenge 2025! "
            "It was the team's first-ever hackathon. The project combines 40 years of NASA POWER satellite data, "
            "SARIMAX machine learning, and Gemini AI to provide accessible future weather forecasting."
        )
        session["conversation_history"] += f"Assistant: {reply}\n"
        return jsonify({"reply": reply})

    forecast_text = ""
    if extracted_location and extracted_date:
        try:
            target_date = pd.to_datetime(extracted_date)
            norm_city = normalize_city_name(extracted_location)
            file_name = f"{norm_city}.json"

            df_t2m, df_tp, has_precip, precip_key_used = load_and_preprocess_data(file_name)
            time_series_temp = df_t2m["T2M"]
            time_series_precip = df_tp[precip_key_used] if has_precip else None

            model_fit_temp = train_model(time_series_temp, cache_key=f"{norm_city}_T2M")
            model_fit_precip = train_model(time_series_precip, cache_key=f"{norm_city}_PRECIP") if has_precip else None
            last_date = time_series_temp.index[-1]

            forecast_context_temp, forecast_context_precip, method = forecast_for_date(
                model_fit_temp, model_fit_precip, last_date, target_date,
                has_precip, precip_key_used, df_t2m=df_t2m, df_tp=df_tp
            )

            temp = float(forecast_context_temp[target_date])
            precip = float(forecast_context_precip[target_date]) if has_precip and forecast_context_precip is not None else 0.0

            if temp >= 35: feeling = "It will feel very hot."
            elif temp >= 28: feeling = "It will feel warm."
            elif temp >= 20: feeling = "It will feel pleasant."
            else: feeling = "It will feel cool."

            if precip >= 10: rain_feel = "Expect heavy rainy conditions."
            elif precip >= 4: rain_feel = "Expect moderate rain."
            elif precip > 0.1: rain_feel = "Light rain is possible."
            else: rain_feel = "It should remain dry."

            mode_label = "Historical NASA Observation" if method == "historical_actual" else "Forecast Prediction"
            forecast_text = (
                f"[{mode_label}] {extracted_location.title()} on {extracted_date}: "
                f"Temperature {round(temp, 2)}°C, Rain: {round(precip, 2)} mm. {feeling} {rain_feel}"
            )
            session["last_forecast"] = {"text": forecast_text, "temp": temp, "precip": precip, "date": extracted_date, "location": extracted_location}
            session["current_location"] = extracted_location
            session["current_date"] = extracted_date
        except Exception as e:
            forecast_text = f"Forecast note: {e}"
            session["last_forecast"] = {"text": forecast_text}

    ai_reply = ""
    if chat_model:
        response_prompt = f"""
        You are the Weather Foreseen AI assistant by Team The Crucibles (built for NASA Space Apps Challenge 2025, our first hackathon!).
        Keep replies concise, friendly, and accurate.
        Conversation history:
        {session['conversation_history']}
        Forecast info: "{forecast_text}"
        Latest query: "{user_input}"
        """
        try:
            response = chat_model.generate_content(response_prompt)
            ai_text = response.text.strip()
            ai_reply = f"{forecast_text}\n{ai_text}".strip() if forecast_text else ai_text
        except Exception:
            ai_reply = forecast_text or "I can provide forecasts for 15+ Indian cities. Try asking: 'Weather in Mumbai on Dec 25'!"
    else:
        ai_reply = forecast_text or "Please provide a city and date (e.g. 'Bengaluru next month')."

    session["conversation_history"] += f"Assistant: {ai_reply}\n"
    return jsonify({"reply": ai_reply})

@app.route("/predict", methods=["POST"])
def predict_weather():
    data = request.get_json() or {}
    city_name = data.get("city", "").strip()
    target_date_str = data.get("date", "").strip()

    if not city_name or not target_date_str:
        return jsonify({"error": "Both City and Date are required."}), 400

    try:
        target_date = pd.to_datetime(target_date_str)
        norm_city = normalize_city_name(city_name)
        file_name = f"{norm_city}.json"

        df_t2m, df_tp, has_precip, precip_key_used = load_and_preprocess_data(file_name)
        time_series_temp = df_t2m["T2M"]
        time_series_precip = df_tp[precip_key_used] if has_precip else None

        model_fit_temp = train_model(time_series_temp, cache_key=f"{norm_city}_T2M")
        model_fit_precip = train_model(time_series_precip, cache_key=f"{norm_city}_PRECIP") if has_precip else None
        last_date = time_series_temp.index[-1]

        forecast_context_temp, forecast_context_precip, method = forecast_for_date(
            model_fit_temp, model_fit_precip, last_date, target_date,
            has_precip, precip_key_used, df_t2m=df_t2m, df_tp=df_tp
        )

        temp = float(forecast_context_temp[target_date])
        precip = float(forecast_context_precip[target_date]) if has_precip and forecast_context_precip is not None else 0.0

        will_rain, rain_details = will_it_rain(precip, temp, target_date, has_precip)

        # Realistic calibrated humidity calculation
        base_humidity = 55 + (precip * 2.5) - ((temp - 24) * 0.7)
        humidity = max(25, min(round(base_humidity, 1), 98))

        # Monotonic rain probability curve
        if precip >= 15.0: rain_probability = "95%"
        elif precip >= 10.0: rain_probability = "85%"
        elif precip >= 5.0: rain_probability = "70%"
        elif precip >= 2.0: rain_probability = "55%"
        elif precip >= 0.5: rain_probability = "40%"
        elif precip > 0: rain_probability = "25%"
        else: rain_probability = "10%"

        method_display = {
            "sarimax_model": "SARIMAX Statistical Model",
            "historical_actual": "Actual NASA Satellite Observation",
            "seasonal_fallback": "Seasonal Trend Baseline"
        }.get(method, "Predictive Model")

        return jsonify({
            "city": norm_city.replace("_", " ").title(),
            "date": target_date.strftime("%Y-%m-%d"),
            "temperature": round(temp, 2),
            "rainfall": round(precip, 2),
            "humidity": humidity,
            "rain_chance": rain_probability,
            "rain_details": rain_details,
            "will_rain": "Yes" if will_rain else "No",
            "pipeline_method": method_display
        })
    except FileNotFoundError as fnf:
        return jsonify({"error": str(fnf)}), 404
    except Exception as e:
        return jsonify({"error": f"Forecasting pipeline note: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(debug=True, port=port)

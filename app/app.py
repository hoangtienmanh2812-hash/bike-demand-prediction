from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/best_model_xgboost.pkl"
TRAIN_PATH = PROJECT_ROOT / "data/processed/train.csv"
MIN_DATE = pd.Timestamp("2011-01-01").date()
MAX_DATE = pd.Timestamp("2012-12-31").date()


@st.cache_resource
def load_model_bundle():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


@st.cache_data
def load_training_defaults():
    train_df = pd.read_csv(TRAIN_PATH)
    return {
        "windspeed": float(train_df["windspeed"].median()),
        "lag_1": float(train_df["lag_1"].median()),
        "lag_2": float(train_df["lag_2"].median()),
        "lag_24": float(train_df["lag_24"].median()),
        "lag_168": float(train_df["lag_168"].median()),
    }


def build_input_row(selected_date, hour, temperature_c, humidity_percent, weather, workingday, holiday, feature_columns):
    timestamp = pd.Timestamp(selected_date) + pd.Timedelta(hours=hour)
    month = timestamp.month
    year = timestamp.year
    day_of_week = timestamp.dayofweek
    season = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}[month]
    normalized_temperature = temperature_c / 41.0
    normalized_humidity = humidity_percent / 100.0
    defaults = load_training_defaults()

    values = {
        "yr": int(year >= 2012),
        "mnth": month,
        "hr": hour,
        "holiday": holiday,
        "weekday": day_of_week,
        "workingday": workingday,
        "season": season,
        "temp": normalized_temperature,
        "atemp": normalized_temperature,
        "hum": normalized_humidity,
        "windspeed": defaults["windspeed"],
        "hour": hour,
        "day": timestamp.day,
        "month": month,
        "year": year,
        "day_of_week": day_of_week,
        "day_of_year": timestamp.dayofyear,
        "is_weekend": int(day_of_week >= 5),
        "is_workingday": workingday,
        "rush_hour": int(hour in [7, 8, 9, 16, 17, 18, 19]),
        "weathersit": weather,
        **{name: defaults[name] for name in ["lag_1", "lag_2", "lag_24", "lag_168"]},
    }
    return pd.DataFrame([[values[column] for column in feature_columns]], columns=feature_columns)


st.set_page_config(page_title="Bike Demand Prediction", page_icon="🚲", layout="centered")
st.title("Bike Demand Prediction")
st.caption("CPU XGBoost model")

if not MODEL_PATH.exists() or not TRAIN_PATH.exists():
    st.error("Model or training data is missing. Run PHASE 11 first.")
    st.stop()

bundle = load_model_bundle()
model = bundle["model"]
feature_columns = bundle["feature_columns"]

selected_date = st.date_input("Date", value=MAX_DATE, min_value=MIN_DATE, max_value=MAX_DATE)
hour = st.slider("Hour", min_value=0, max_value=23, value=8)
temperature_c = st.number_input("Temperature (°C)", min_value=-20.0, max_value=45.0, value=20.0, step=0.5)
humidity_percent = st.slider("Humidity (%)", min_value=0, max_value=100, value=60)
weather = st.selectbox(
    "Weather",
    options=[1, 2, 3, 4],
    format_func=lambda value: {
        1: "Clear / partly cloudy",
        2: "Mist / cloudy",
        3: "Light rain or snow",
        4: "Heavy rain or snow",
    }[value],
)
workingday = int(st.checkbox("Working day", value=True))
holiday = int(st.checkbox("Holiday", value=False))

if st.button("Predict demand", type="primary"):
    input_row = build_input_row(
        selected_date,
        hour,
        temperature_c,
        humidity_percent,
        weather,
        workingday,
        holiday,
        feature_columns,
    )
    prediction = max(0.0, float(model.predict(input_row)[0]))
    st.metric("Predicted bike rentals", f"{prediction:,.0f}")
    st.caption("Lag and windspeed inputs use training-set medians because they are not entered in this simple interface.")

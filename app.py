import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Health Prediction App", page_icon="🧠", layout="wide")

# --- Load Models ---
try:
    sleep_pipe = joblib.load("dt1.pkl")
    stress_pipe = joblib.load("dt2.pkl")
    st.sidebar.success("✅ Models loaded successfully")
except:
    st.sidebar.error("❌ Could not load models. Please check file paths.")
    sleep_pipe, stress_pipe = None, None

# --- Sidebar Info ---
st.sidebar.title("ℹ️ About App")
st.sidebar.info(
    """
    🧠 **Health Prediction App**  
    This app uses Machine Learning models to predict:  
    - 😴 **Sleep Quality**  
    - ⚡ **Stress Level**  
    based on your lifestyle habits.  
    """
)

# --- Feature Order ---
feature_order = [
    "Age", "Caffeine_mg", "Sleep_Hours", "BMI",
    "Heart_Rate", "Physical_Activity_Hours"
]

def to_row(payload):
    row = []
    for key in feature_order:
        value = payload.get(key, 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        row.append(value)
    return row

# --- Main UI ---
st.title("🧠 Health Prediction App")
st.write("Fill in your details below to predict your **Sleep Quality** and **Stress Level**.")

st.markdown("---")

# --- User Inputs ---
st.subheader("📌 Enter Your Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=0, max_value=120, step=1)
    caffeine = st.number_input("☕ Caffeine Intake (mg/day)", min_value=0, step=10)
    sleep_hours = st.number_input("🛌 Sleep Hours (per night)", min_value=0.0, step=0.5)

with col2:
    bmi = st.number_input("⚖️ BMI", min_value=0.0, step=0.1)
    heart_rate = st.number_input("❤️ Heart Rate (bpm)", min_value=0, step=1)
    activity = st.number_input("🏃 Physical Activity Hours (per week)", min_value=0.0, step=0.5)

input_data = {
    "Age": age,
    "Caffeine_mg": caffeine,
    "Sleep_Hours": sleep_hours,
    "BMI": bmi,
    "Heart_Rate": heart_rate,
    "Physical_Activity_Hours": activity
}

# --- Prediction ---
st.markdown("---")
if st.button("🔮 Predict My Health"):
    if sleep_pipe is not None and stress_pipe is not None:
        row = [to_row(input_data)]

        sleep_pred = sleep_pipe.predict(row)[0]
        stress_pred = stress_pipe.predict(row)[0]

        st.markdown("## 📊 Prediction Results")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("😴 Sleep Quality", f"{sleep_pred}")
        with col2:
            st.metric("⚡ Stress Level", f"{stress_pred}")

        st.success("✅ Prediction completed successfully!")
    else:
        st.error("❌ No trained models available. Please train and save them first.")

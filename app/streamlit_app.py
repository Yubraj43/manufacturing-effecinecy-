import streamlit as st
import joblib
import pickle
from pathlib import Path
import numpy as np
import streamlit as st

st.set_page_config(page_title="Manufacturing Efficiency AI")
st.title("AI Manufacturing Efficiency Predictor")

# Project root: ...\Manufacturing_Efficiency_AI
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "efficiency_model.pkl"

st.write("Project root:", str(BASE_DIR))
st.write("Model path:", str(MODEL_PATH))

# 1) Check folder/file existence first
if not MODELS_DIR.exists():
    st.error(f"'models' folder not found: {MODELS_DIR}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info(f"Files in models/: {[p.name for p in MODELS_DIR.iterdir()]}")
    st.stop()

if MODEL_PATH.stat().st_size == 0:
    st.error("Model file is empty (0 bytes). Re-save the model.")
    st.stop()

# 2) Try loading with joblib, then pickle fallback
model = None
load_error = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    load_error = e
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception:
        pass

if model is None:
    st.error(f"Model load failed: {type(load_error).__name__}: {load_error}")
    st.info("Common cause: scikit-learn version mismatch between training and runtime.")
    st.info("Install matching versions from requirements.txt and retry.")
    st.stop()

st.success("Model loaded successfully ✅")


st.header("Enter Machine Sensor Data")

# Encoded numeric inputs (because notebook label-encoded these columns)
machine_id = st.number_input("Machine_ID (encoded)", min_value=0, value=0, step=1)
operation_mode = st.number_input("Operation_Mode (encoded)", min_value=0, value=0, step=1)

temperature = st.slider("Temperature_C", 20, 120, 60)
vibration = st.slider("Vibration_Hz", 0, 100, 30)
power = st.slider("Power_Consumption_kW", 0, 200, 80)
latency = st.slider("Network_Latency_ms", 0, 100, 20)
packet_loss = st.slider("Packet_Loss_%", 0.0, 10.0, 1.0)
defect_rate = st.slider("Defect_Rate_%", 0.0, 10.0, 1.0)
speed = st.slider("Production_Speed_units_per_hr", 0, 500, 200)
maintenance = st.slider("Predictive_Maintenance_Score", 0, 100, 50)
error_rate = st.slider("Error_Rate_%", 0.0, 10.0, 1.0)

# Engineered features used during training
error_output_ratio = float(error_rate) / float(speed) if speed != 0 else 0.0
network_reliability = 100.0 - float(packet_loss) - (float(latency) / 10.0)
energy_efficiency = float(speed) / float(power) if power != 0 else 0.0

# Exact 14-feature order from notebook training
input_data = np.array([[
    machine_id,
    operation_mode,
    temperature,
    vibration,
    power,
    latency,
    packet_loss,
    defect_rate,
    speed,
    maintenance,
    error_rate,
    error_output_ratio,
    network_reliability,
    energy_efficiency
]], dtype=float)

if st.button("Predict Efficiency"):
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and input_data.shape[1] != expected:
        st.error(f"Model expects {expected} features, got {input_data.shape[1]}")
    else:
        prediction = model.predict(input_data)
        st.success(f"Prediction: {prediction[0]}")
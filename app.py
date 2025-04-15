import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler

# ---- Load trained model and scaler ----
try:
    model = load("stacking_model.pkl")  # Ensure this file exists
    scaler = load("scaler.pkl")  # Ensure this file exists
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Error loading model or scaler: {e}")
    model_loaded = False

# ---- Streamlit UI ----
st.title("🔬 Breast Cancer Prediction Model")
st.markdown("This model predicts whether a tumor is **Benign (Non-Cancerous) or Malignant (Cancerous)** based on cell sample features.")

st.sidebar.header("📝 Enter Patient Data")

# Define feature names (must match those used in training)
feature_names = [
    "mean radius", "mean texture", "mean area", "mean concavity", "mean concave points",
    "mean fractal dimension", "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error", "concave points error", "symmetry error",
    "fractal dimension error", "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity", "worst concave points", "worst symmetry",
    "worst fractal dimension"
]

# Create input fields for all features with 4 decimal places
user_input = {}
for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(
        feature.replace("_", " ").title(), min_value=0.0, format="%.4f"
    )

# When user clicks "Predict"
if st.sidebar.button("🔍 Predict") and model_loaded:
    # Convert user input to DataFrame
    input_data = pd.DataFrame([list(user_input.values())], columns=feature_names)

    # Ensure input data matches the trained model's feature set
    if set(input_data.columns) != set(feature_names):
        st.error("⚠️ Feature mismatch! Please check input fields.")
    else:
        # Scale the input data using the same scaler from training
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Display result
        if prediction[0] == 1:
            st.success("🔴 Prediction: **Malignant (Cancerous)**")
        else:
            st.success("🟢 Prediction: **Benign (Non-Cancerous)**")

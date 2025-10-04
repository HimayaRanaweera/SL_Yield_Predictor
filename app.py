# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
import sklearn

st.set_page_config(page_title="SL Yield Predictor", page_icon="🌾", layout="centered")

@st.cache_resource
def load_model():
    try:
        # Register the missing class before loading
        import sys
        module = sys.modules['sklearn.compose._column_transformer']
        if not hasattr(module, '_RemainderColsList'):
            setattr(module, '_RemainderColsList', _RemainderColsList)
        
        model = joblib.load("best_model_tuned.joblib")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title("🌾 Sri Lanka Crop Yield Predictor")
st.caption("Predict expected yield (mt/ha) before the season ends and estimate production.")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=2010, max_value=2030, value=2024)
        season = st.selectbox("Season", ["Maha","Yala"])
        province = st.selectbox("Province", ["Western","Central","Southern","Northern","Eastern",
                                             "North Western","North Central","Uva","Sabaragamuwa"])
        district = st.text_input("District (e.g., Kandy)")
        crop = st.selectbox("Crop", ["Paddy","Maize","Vegetables","Onion","Chili","Coconut","Rubber","Tea"])
    with col2:
        soil = st.selectbox("Soil_Type", ["Clay","Sandy","Loamy","Laterite","Alluvial"])
        irrigation = st.selectbox("Irrigation", ["Irrigated","Rainfed"])
        area_sown = st.number_input("Area_Sown_ha", min_value=0.0, value=100.0, step=1.0)
        rain = st.number_input("Rainfall_mm", min_value=0.0, value=1500.0, step=10.0)
        temp = st.number_input("Temperature_C", min_value=0.0, value=28.0, step=0.1)
        fert = st.number_input("Fertilizer_kg_per_ha", min_value=0.0, value=220.0, step=1.0)
        price = st.number_input("Market_Price_LKR_per_kg", min_value=0.0, value=120.0, step=1.0)

    submitted = st.form_submit_button("Predict Yield")

if submitted and model is not None:
    try:
        row = pd.DataFrame([{
            "Year": year, "Season": season, "Province": province, "District": district,
            "Crop": crop, "Soil_Type": soil, "Irrigation": irrigation,
            "Area_Sown_ha": area_sown, "Rainfall_mm": rain, "Temperature_C": temp,
            "Fertilizer_kg_per_ha": fert, "Market_Price_LKR_per_kg": price
        }])
        pred = model.predict(row)[0]
        st.success(f"Predicted Yield: {pred:.2f} mt/ha")
        st.info(f"Estimated Production: {pred * area_sown:.1f} metric tons")
    except Exception as e:
        st.error(f"Prediction error: {e}")

if model is None:
    st.warning("Model not loaded. Please check if 'best_model_tuned.joblib' exists in the app directory.")

st.caption("Model: Tuned GradientBoostingRegressor")

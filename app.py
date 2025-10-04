# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json

st.set_page_config(page_title="SL Yield Predictor", page_icon="🌾", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("yield_model.joblib")
    try:
        with open("yield_model_meta.json") as f:
            meta = json.load(f)
    except Exception:
        meta = {}
    return model, meta

model, meta = load_artifacts()

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

if submitted:
    row = pd.DataFrame([{
        "Year": year, "Season": season, "Province": province, "District": district,
        "Crop": crop, "Soil_Type": soil, "Irrigation": irrigation,
        "Area_Sown_ha": area_sown, "Rainfall_mm": rain, "Temperature_C": temp,
        "Fertilizer_kg_per_ha": fert, "Market_Price_LKR_per_kg": price
    }])
    pred = model.predict(row)[0]
    st.success(f"Predicted Yield: {pred:.2f} mt/ha")
    st.info(f"Estimated Production: {pred * area_sown:.1f} metric tons")

st.caption(f"Model: {meta.get('best_model','GradientBoostingRegressor')} • "
           f"RMSE: {meta.get('rmse','-')} • R²: {meta.get('r2','-')}")'''

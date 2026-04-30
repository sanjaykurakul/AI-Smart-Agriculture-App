import streamlit as st
import pandas as pd
from crop_recommendation import CropModel

# simple page config
st.set_page_config(page_title="Crop Predictor", layout="centered")

st.title("🌾 Crop Recommendation System")
st.write("Fill in the soil and weather conditions below to find the best crop to plant.")

@st.cache_resource
def load_my_model():
    m = CropModel()
    X_train, X_test, y_train, y_test = m.load_data()
    m.train(X_train, y_train)
    return m

model = load_my_model()

st.divider()

# Organize inputs into two columns so it looks neater
col1, col2 = st.columns(2)

with col1:
    st.subheader("Soil Nutrients")
    n = st.number_input("Nitrogen (N)", value=90)
    p = st.number_input("Phosphorus (P)", value=42)
    k = st.number_input("Potassium (K)", value=43)
    ph = st.number_input("pH Level", value=6.5)

with col2:
    st.subheader("Weather Conditions")
    temp = st.number_input("Temperature (°C)", value=20.8)
    hum = st.number_input("Humidity (%)", value=82.0)
    rain = st.number_input("Rainfall (mm)", value=202.9)

st.write("") # empty space

if st.button("Predict Best Crop", type="primary", use_container_width=True):
    data = {
        'N': n,
        'P': p,
        'K': k,
        'temperature': temp,
        'humidity': hum,
        'ph': ph,
        'rainfall': rain
    }
    
    crop = model.predict(data)
    profit = model.get_profit(crop)
    
    st.markdown("### Results:")
    st.success("✅ **Recommended Crop:** " + crop.capitalize())
    st.info("💰 **Estimated Profit:** ₹" + str(profit) + " per acre")

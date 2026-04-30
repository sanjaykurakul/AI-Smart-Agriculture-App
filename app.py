import streamlit as st
import pandas as pd
from crop_recommendation import CropModel

st.title("Crop Recommendation System")

@st.cache_resource
def load_my_model():
    m = CropModel()
    X_train, X_test, y_train, y_test = m.load_data()
    m.train(X_train, y_train)
    return m

model = load_my_model()

st.write("Enter the soil and weather details to get the best crop.")

n = st.number_input("Nitrogen (N)", value=90)
p = st.number_input("Phosphorus (P)", value=42)
k = st.number_input("Potassium (K)", value=43)
temp = st.number_input("Temperature", value=20.8)
hum = st.number_input("Humidity", value=82.0)
ph = st.number_input("pH Level", value=6.5)
rain = st.number_input("Rainfall", value=202.9)

if st.button("Predict"):
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
    
    st.success("Recommended Crop: " + crop.upper())
    st.info("Estimated Profit: ₹" + str(profit) + " per acre")

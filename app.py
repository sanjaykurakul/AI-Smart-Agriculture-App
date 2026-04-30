import streamlit as st
import pandas as pd
from crop_recommendation import CropModel

st.set_page_config(page_title="Crop Predictor", layout="centered")

st.title("🌾 Crop Recommendation System")
st.write("Fill in the soil and weather conditions below to find the best crop.")

@st.cache_resource
def load_model():
    m = CropModel()
    X_train, X_test, y_train, y_test = m.load_data()
    m.train(X_train, y_train)
    return m

model = load_model()

st.divider()

# INPUTS
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

st.write("")

# 🔥 PREDICT BUTTON (FIXED)
if st.button("🚀 Predict Best Crop", use_container_width=True):

    data = {
        'N': n,
        'P': p,
        'K': k,
        'temperature': temp,
        'humidity': hum,
        'ph': ph,
        'rainfall': rain
    }

    crop, confidence = model.predict_with_confidence(data)
    profit = model.get_profit(crop)
    explanation = model.explain_prediction(data)

    st.markdown("## 🌾 Prediction Results")

    st.success(f"✅ Recommended Crop: {crop.upper()}")
    st.info(f"💰 Estimated Profit: ₹{profit} per acre")
    st.warning(f"📊 Confidence: {confidence}%")

    st.markdown("### 🤖 AI Explanation")
    st.write(explanation)

    st.markdown("### 📊 Feature Importance")
    st.bar_chart(model.model.feature_importances_)

# ---------------- CHATBOT ----------------

st.divider()
st.markdown("## 🤖 Farmer Assistant Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask your question:")

if st.button("Send"):
    if user_input:
        response = model.chatbot_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

# DISPLAY CHAT
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.write(f"🧑‍🌾 {msg}")
    else:
        st.write(f"🤖 {msg}")

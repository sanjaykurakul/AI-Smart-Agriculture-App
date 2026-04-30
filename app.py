import streamlit as st
import pandas as pd
import os
from crop_recommendation import CropModel

# Page config
st.set_page_config(page_title="AI Crop Predictor", layout="centered")

st.title("🌾 AI Crop Recommendation System")
st.write("Enter soil and weather conditions to get the best crop recommendation.")

# Load model
@st.cache_resource
def load_model():
    m = CropModel()
    X_train, X_test, y_train, y_test = m.load_data()
    m.train(X_train, y_train)
    return m

model = load_model()

st.divider()

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Soil Nutrients")
    n = st.number_input("Nitrogen (N)", value=90)
    p = st.number_input("Phosphorus (P)", value=42)
    k = st.number_input("Potassium (K)", value=43)
    ph = st.number_input("pH Level", value=6.5)

with col2:
    st.subheader("Weather Conditions")
    temp = st.number_input("Temperature (°C)", value=25.0)
    hum = st.number_input("Humidity (%)", value=80.0)
    rain = st.number_input("Rainfall (mm)", value=200.0)

st.write("")

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Best Crop", use_container_width=True):

    try:
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

        # ---------------- IMAGE ----------------
        image_path = f"images/{crop}.jpg"

        if os.path.exists(image_path):
            st.image(image_path, caption=crop.upper(), use_container_width=True)
        else:
            st.warning("Image not available for this crop.")

    except Exception as e:
        st.error(f"Error: {e}")

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

# Display chat
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.write(f"🧑‍🌾 {msg}")
    else:
        st.write(f"🤖 {msg}")

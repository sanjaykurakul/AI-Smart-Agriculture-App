import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class CropModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        self.profit_data = {
            'rice': 64000, 'maize': 52000, 'chickpea': 56000,
            'kidneybeans': 60000, 'pigeonpeas': 57600,
            'mothbeans': 54400, 'mungbean': 55200,
            'blackgram': 56800, 'lentil': 58400,
            'pomegranate': 96000, 'banana': 72000,
            'mango': 120000, 'grapes': 160000,
            'watermelon': 88000, 'melon': 84000,
            'apple': 200000, 'orange': 144000,
            'papaya': 104000, 'coconut': 128000,
            'cotton': 76000, 'jute': 48000,
            'coffee': 176000
        }

    def get_dummy_data(self, n=2000):
        np.random.seed(42)
        crops = list(self.profit_data.keys())

        data = {
            'N': np.random.randint(0, 140, n),
            'P': np.random.randint(5, 145, n),
            'K': np.random.randint(5, 205, n),
            'temperature': np.random.uniform(8.0, 45.0, n),
            'humidity': np.random.uniform(14.0, 100.0, n),
            'ph': np.random.uniform(3.5, 9.9, n),
            'rainfall': np.random.uniform(20.0, 300.0, n),
            'label': np.random.choice(crops, n)
        }
        return pd.DataFrame(data)

    def load_data(self):
        try:
            df = pd.read_csv("Crop_recommendation.csv")
        except:
            df = self.get_dummy_data()

        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_with_confidence(self, features):
        df = pd.DataFrame([features])
        prediction = self.model.predict(df)[0]
        confidence = max(self.model.predict_proba(df)[0])
        return prediction, round(confidence * 100, 2)

    def get_profit(self, crop):
        return self.profit_data.get(crop, 0)

    def explain_prediction(self, features):
        explanations = []

        if features['rainfall'] > 200:
            explanations.append("High rainfall supports water-intensive crops.")
        if features['temperature'] > 30:
            explanations.append("Warm temperature favors tropical crops.")
        if features['ph'] < 5.5:
            explanations.append("Acidic soil affects crop suitability.")
        if features['humidity'] > 70:
            explanations.append("High humidity supports moisture-loving crops.")

        if not explanations:
            explanations.append("Conditions are balanced for general crop growth.")

        return " ".join(explanations)

    def chatbot_response(self, user_query):
        query = user_query.lower()

        if "best crop" in query:
            return "Use the predictor above to find the best crop."
        elif "rice" in query:
            return "Rice grows well in high rainfall and humid conditions."
        elif "temperature" in query:
            return "Most crops prefer temperatures between 20°C to 30°C."
        elif "profit" in query:
            return "Mango, grapes, and apple are high-profit crops."
        elif "fertilizer" in query:
            return "Balanced NPK fertilizers improve soil fertility."
        else:
            return "Please provide more details about your conditions."

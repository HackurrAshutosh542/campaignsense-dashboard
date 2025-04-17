import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("xgboost_best_model.pkl")

# Title for Dashboard
st.title("🚀 AI Marketing Campaign Optimizer")

# User input features (modify these according to your actual model features)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Button for Prediction
if st.button("Predict Campaign Success"):
    input_features = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_features)[0]
    st.success(f"🎯 Predicted Campaign Success: {int(prediction)}%")

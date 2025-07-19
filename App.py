# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
model = joblib.load('attrition_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title("Employee Attrition Predictor")

st.markdown("🔍 Enter employee details to predict if they'll leave the company.")

# Input fields
def user_input():
    Age = st.slider('Age', 18, 60, 30)
    DistanceFromHome = st.slider('Distance From Home', 1, 30, 10)
    MonthlyIncome = st.slider('Monthly Income', 1000, 20000, 5000)
    JobSatisfaction = st.slider('Job Satisfaction', 1, 4, 2)
    EnvironmentSatisfaction = st.slider('Environment Satisfaction', 1, 4, 2)
    YearsAtCompany = st.slider('Years at Company', 0, 40, 5)
    WorkLifeBalance = st.slider('Work Life Balance', 1, 4, 2)
    OverTime = st.selectbox("OverTime", ['Yes', 'No'])

    # Convert OverTime to binary
    OverTime = 1 if OverTime == 'Yes' else 0

    data = {
        'Age': Age,
        'DistanceFromHome': DistanceFromHome,
        'MonthlyIncome': MonthlyIncome,
        'JobSatisfaction': JobSatisfaction,
        'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'YearsAtCompany': YearsAtCompany,
        'WorkLifeBalance': WorkLifeBalance,
        'OverTime': OverTime
    }

    features = pd.DataFrame([data])
    return features

input_df = user_input()
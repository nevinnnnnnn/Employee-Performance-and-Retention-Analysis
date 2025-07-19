import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('attrition_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("Employee Attrition Prediction App")

# Input form
st.header("Enter Employee Details")

age = st.slider("Age", 18, 60, 30)
distance = st.slider("Distance From Home (in km)", 1, 30, 5)
income = st.number_input("Monthly Income (₹)", min_value=1000, max_value=100000, value=30000, step=500)
job_satisfaction = st.selectbox("Job Satisfaction (1 = Low, 4 = High)", [1, 2, 3, 4])
env_satisfaction = st.selectbox("Environment Satisfaction (1 = Low, 4 = High)", [1, 2, 3, 4])
years_at_company = st.slider("Years at Company", 0, 40, 5)
work_life_balance = st.selectbox("Work Life Balance (1 = Bad, 4 = Excellent)", [1, 2, 3, 4])
overtime = st.selectbox("OverTime", ["No", "Yes"])

# Convert OverTime to binary
overtime_binary = 1 if overtime == "Yes" else 0

# Prediction button
if st.button("Predict Attrition"):
    input_data = pd.DataFrame([{
        "Age": age,
        "DistanceFromHome": distance,
        "MonthlyIncome": income,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": env_satisfaction,
        "YearsAtCompany": years_at_company,
        "WorkLifeBalance": work_life_balance,
        "OverTime": overtime_binary
    }])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction[0] == 1:
        st.error(f"⚠️ The employee is likely to leave the company. (Attrition Probability: {prob:.2f})")
    else:
        st.success(f"✅ The employee is likely to stay. (Attrition Probability: {prob:.2f})")

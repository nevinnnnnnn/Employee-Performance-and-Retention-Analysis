import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('attrition_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title and Description
st.title("💼 Employee Attrition Predictor")
st.markdown("🔍 Enter employee details to predict if they're likely to leave the company.")

# Function to collect user input
def user_input():
    Age = st.slider('🎂 Age', 18, 60, 30)
    DistanceFromHome = st.slider('📍 Distance From Home (in km)', 1, 30, 10)
    MonthlyIncome = st.slider('💰 Monthly Income (₹)', 1000, 20000, 5000)
    JobSatisfaction = st.slider('🙂 Job Satisfaction (1 - Low, 4 - High)', 1, 4, 2)
    EnvironmentSatisfaction = st.slider('🏢 Environment Satisfaction (1 - Low, 4 - High)', 1, 4, 2)
    YearsAtCompany = st.slider('📅 Years at Company', 0, 40, 5)
    WorkLifeBalance = st.slider('⚖️ Work-Life Balance (1 - Bad, 4 - Best)', 1, 4, 2)
    OverTime = st.selectbox("⏰ OverTime", ['Yes', 'No'])

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

    return pd.DataFrame([data])

# Get user input
input_df = user_input()

# Predict only if button is clicked
if st.button("📊 Predict"):
    # Scale input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display result
    st.subheader("🎯 Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ The employee is **likely to leave** the company.")
    else:
        st.success("✅ The employee is **likely to stay** with the company.")

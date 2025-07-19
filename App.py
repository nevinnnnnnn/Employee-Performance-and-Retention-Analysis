# app.py

import streamlit as st
import joblib
import pandas as pd

# Load model, scaler, and expected column order
try:
    model = joblib.load('attrition_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')  # List of expected columns
except Exception as e:
    st.error(f"🚨 Error loading model, scaler, or columns: {e}")
    st.stop()

# Title and Description
st.title("💼 Employee Attrition Predictor")
st.markdown("🔍 Enter employee details to predict if they're likely to leave the company.")

# Function to collect user input
def user_input():
    # Numeric Inputs
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

    # Categorical Inputs
    BusinessTravel = st.selectbox('🧳 Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
    DailyRate = st.slider('📊 Daily Rate', 100, 1500, 500)
    Department = st.selectbox('🏢 Department', ['Sales', 'Research & Development', 'Human Resources'])
    Education = st.slider('🎓 Education (1-Below College, 5-Doctor)', 1, 5, 3)
    EducationField = st.selectbox('📚 Education Field', ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
    Gender = st.selectbox('👤 Gender', ['Male', 'Female'])
    JobRole = st.selectbox('💼 Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                                          'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
    MaritalStatus = st.selectbox('💍 Marital Status', ['Single', 'Married', 'Divorced'])

    # Create DataFrame
    data = {
        'Age': Age,
        'DistanceFromHome': DistanceFromHome,
        'MonthlyIncome': MonthlyIncome,
        'JobSatisfaction': JobSatisfaction,
        'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'YearsAtCompany': YearsAtCompany,
        'WorkLifeBalance': WorkLifeBalance,
        'OverTime': OverTime,
        'BusinessTravel': BusinessTravel,
        'DailyRate': DailyRate,
        'Department': Department,
        'Education': Education,
        'EducationField': EducationField,
        'Gender': Gender,
        'JobRole': JobRole,
        'MaritalStatus': MaritalStatus
    }

    return pd.DataFrame([data])

# Collect input
input_df = user_input()

# Show raw input
st.subheader("🧾 Input Summary")
st.write(input_df)

# Preprocess input
try:
    # One-hot encode categorical columns
    input_encoded = pd.get_dummies(input_df)

    # Add missing columns from training
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Ensure column order matches training
    input_encoded = input_encoded[model_columns]

    # Show preprocessed input (optional)
    # st.write("Preprocessed Input:")
    # st.write(input_encoded)

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict on button click
    if st.button("📊 Predict"):
        prediction = model.predict(input_scaled)
        st.subheader("🎯 Prediction Result")
        if prediction[0] == 1:
            st.error("⚠️ The employee is **likely to leave** the company.")
        else:
            st.success("✅ The employee is **likely to stay** with the company.")

except Exception as e:
    st.error(f"🚨 Prediction error: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load Model and Scaler ---
# Load the pre-trained logistic regression model
try:
    with open('Logistic_Regression_heart.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file 'Logistic_Regression_heart.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Load the scaler that was fitted on the training data
try:
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Scaler file 'scaler.pkl' not found. Please create it from your notebook and place it in the same directory.")
    st.stop()


# --- App Title and Description ---
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction Application")
st.markdown("""
This application uses a Logistic Regression model to predict the likelihood of a patient having heart disease based on their clinical data.
Please enter the patient's information in the fields below.
""")

# --- User Input Section ---
st.header("Patient Data Input")

# Create columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=54, step=1)
    sex = st.selectbox("Sex", ("Male", "Female"))
    chest_pain_type = st.selectbox("Chest Pain Type", ("Typical Angina (TA)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"))
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("No", "Yes"))

with col2:
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=132, step=1)
    cholesterol = st.number_input("Cholesterol (mm/dl)", min_value=100, max_value=600, value=244, step=1)
    resting_ecg = st.selectbox("Resting ECG", ("Normal", "ST-T wave abnormality (ST)", "Left ventricular hypertrophy (LVH)"))

with col3:
    max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=136, step=1)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ("No", "Yes"))
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=0.8, step=0.1)
    st_slope = st.selectbox("ST Slope", ("Up", "Flat", "Down"))


# --- Prediction Logic ---
if st.button("Predict Heart Disease", use_container_width=True):
    # 1. Create a DataFrame from user inputs
    input_data = {
        'Age': [age],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [1 if fasting_bs == "Yes" else 0],
        'MaxHR': [max_hr],
        'Oldpeak': [oldpeak],
        'Sex_M': [1 if sex == "Male" else 0],
        'ChestPainType_ATA': [1 if chest_pain_type == "Atypical Angina (ATA)" else 0],
        'ChestPainType_NAP': [1 if chest_pain_type == "Non-Anginal Pain (NAP)" else 0],
        'ChestPainType_TA': [1 if chest_pain_type == "Typical Angina (TA)" else 0],
        'RestingECG_Normal': [1 if resting_ecg == "Normal" else 0],
        'RestingECG_ST': [1 if resting_ecg == "ST-T wave abnormality (ST)" else 0],
        'ExerciseAngina_Y': [1 if exercise_angina == "Yes" else 0],
        'ST_Slope_Flat': [1 if st_slope == "Flat" else 0],
        'ST_Slope_Up': [1 if st_slope == "Up" else 0]
    }
    input_df = pd.DataFrame(input_data)

    # Reorder columns to match the model's training order
    # This is a crucial step!
    expected_columns = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    input_df = input_df[expected_columns]

    # 2. Scale the numerical features using the loaded scaler
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # 3. Make a prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # --- Display Results ---
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"**The model predicts a HIGH probability of Heart Disease.**")
        st.write(f"Confidence: **{prediction_proba[0][1]*100:.2f}%**")
        st.warning("This is a predictive model and not a substitute for professional medical advice. Please consult a doctor.")
    else:
        st.success(f"**The model predicts a LOW probability of Heart Disease.**")
        st.write(f"Confidence: **{prediction_proba[0][0]*100:.2f}%**")
        st.info("This is a predictive model and not a substitute for professional medical advice.")

    st.write("---")
    st.write("### Input Data Overview")
    st.dataframe(pd.DataFrame(input_data, index=["Patient Data"]))

import streamlit as st
import joblib
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # Change this if the model file name is different

model = load_model()

# Streamlit UI
st.title("Heart Disease Prediction")

# User Input Fields
age = st.number_input("Age", min_value=1, step=1)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
chest_pain = st.number_input("Chest pain type (1-4)", min_value=1, max_value=4, step=1)
bp = st.number_input("Blood Pressure", min_value=0, step=1)
cholesterol = st.number_input("Cholesterol", min_value=0, step=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 (0 = No, 1 = Yes)", [0, 1])
ekg = st.number_input("EKG Results", min_value=0, step=1)
max_hr = st.number_input("Max Heart Rate", min_value=0, step=1)
exercise_angina = st.selectbox("Exercise Angina (0 = No, 1 = Yes)", [0, 1])
st_depression = st.number_input("ST Depression", min_value=0.0, step=0.1)
slope_st = st.number_input("Slope of ST (1-3)", min_value=1, max_value=3, step=1)
num_vessels = st.number_input("Number of vessels fluro", min_value=0, step=1)
thallium = st.number_input("Thallium (1-3)", min_value=1, max_value=3, step=1)

# Convert input to numpy array for prediction
input_data = np.array([[age, sex, chest_pain, bp, cholesterol, fbs, ekg, max_hr, exercise_angina, 
                         st_depression, slope_st, num_vessels, thallium]])

# Prediction Button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]  # 0: No disease, 1: Disease
    result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
    st.subheader(f"Prediction: {result}")

import streamlit as st
import pickle
import numpy as np

def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def main():
    st.title("ML Model Deployment using Streamlit")
    
    st.write("Enter the feature values for prediction:")
    
    # Example input fields (modify according to your model features)
    feature1 = st.number_input("Age", value=0.0)
    feature2 = st.number_input("Sex", value=0.0)
    feature3 = st.number_input("Chest pain type", value=0.0)
    feature4 = st.number_input("Blood Pressure (BP)", value=0.0)
    feature5 = st.number_input("Cholesterol", value=0.0)
    feature6 = st.number_input("FBS over 120", value=0.0)
    feature7 = st.number_input("EKG results", value=0.0)
    feature8 = st.number_input("Max Heart Rate (HR)", value=0.0)
    feature9 = st.number_input("Exercise-induced angina", value=0.0)
    feature10 = st.number_input("ST depression", value=0.0)
    feature11 = st.number_input("Slope of ST", value=0.0)
    feature12 = st.number_input("Number of vessels fluro", value=0.0)
    feature13 = st.number_input("Thallium", value=0.0)
    
    model = load_model()
    
    if st.button("Predict"):
        input_data = np.array([[feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10,feature11,feature12,feature13]])
        prediction = model.predict(input_data)
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()

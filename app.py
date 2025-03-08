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
    feature1 = st.number_input("Petrol Tax: ", value=0.0)
    feature2 = st.number_input("Average income", value=0.0)
    feature3 = st.number_input("Paved Highways", value=0.0)
    feature4 = st.number_input("Population_Driver_licence(%)", value=0.0)
    
    model = load_model()
    
    if st.button("Predict"):
        input_data = np.array([[feature1, feature2, feature3,feature4]])
        prediction = model.predict(input_data)
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()

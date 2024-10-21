import streamlit as st
import pickle
import numpy as np

# Load your trained logistic regression model
with open('model2.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.title("Heart Disease Prediction")

# Input fields for user data
age = st.number_input("Age", min_value=0, max_value=120, value=25)
sex = st.selectbox("Sex", [0, 1])  # 0 for Female, 1 for Male
chest_pain = st.selectbox("Chest Pain Type", [0, 1, 2, 3])  # Example options
bp = st.number_input("Blood Pressure", min_value=0, value=120)
cholesterol = st.number_input("Cholesterol", min_value=0, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])  # 0 = False, 1 = True
ekg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])  # Example options
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])  # 0 = No, 1 = Yes
st_depression = st.number_input("ST Depression Induced by Exercise Relative to Rest", value=0.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])  # Example options
vessels_fluro = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", [0, 1, 2, 3])
thallium = st.selectbox("Thallium Stress Test Result", [0, 1, 2])  # Example options

# Create a DataFrame for input features
input_features = np.array([[age, sex, chest_pain, bp, cholesterol, fbs, ekg, max_hr,
                            exercise_angina, st_depression, slope, vessels_fluro, thallium]])

# Button for prediction
if st.button("Predict"):
    # Predict the result
    prediction = model.predict(input_features)

    # Convert prediction output to a human-readable form
    result = 'Presence of heart disease' if prediction[0] == 1 else 'Absence of heart disease'

    # Display the result
    st.success(result)

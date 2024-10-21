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
sex = st.selectbox("Sex", ["Female (0)", "Male (1)"])  # More descriptive options
chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4])  # Direct integer values
bp = st.number_input("Blood Pressure (in mm Hg)", min_value=0, value=120)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])  # More descriptive options
ekg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])  # Direct integer values
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
st_depression = st.number_input("ST Depression Induced by Exercise Relative to Rest", value=0.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])  # Direct integer values
vessels_fluro = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", [0, 1, 2, 3])
thallium = st.selectbox("Thallium Stress Test Result", [0, 1, 2])  # Direct integer values

# Create a DataFrame for input features
input_features = np.array([[age,
                            1 if "Male" in sex else 0,
                            chest_pain,
                            bp,
                            cholesterol,
                            1 if "Yes" in fbs else 0,
                            ekg,
                            max_hr,
                            1 if "Yes" in exercise_angina else 0,
                            st_depression,
                            slope,
                            vessels_fluro,
                            thallium]])

# Button for prediction
if st.button("Predict"):
    # Predict the result
    prediction = model.predict(input_features)

    # Convert prediction output to a human-readable form
    result = 'Presence of heart disease' if prediction[0] == 1 else 'Absence of heart disease'

    # Display the result
    st.success(result)

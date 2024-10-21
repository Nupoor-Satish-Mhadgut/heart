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
chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina (1)", "Atypical Angina (2)", "Non-Anginal Pain (3)", "Asymptomatic (4)"])
bp = st.number_input("Blood Pressure (in mm Hg)", min_value=0, value=120)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])  # More descriptive options
ekg = st.selectbox("Resting Electrocardiographic Results", ["Normal (0)", "Having ST-T Wave Abnormality (1)", "Showing Left Ventricular Hypertrophy (2)"])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
st_depression = st.number_input("ST Depression Induced by Exercise Relative to Rest", value=0.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upward Sloping (0)", "Flat (1)", "Downward Sloping (2)"])
vessels_fluro = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", [0, 1, 2, 3])
thallium = st.selectbox("Thallium Stress Test Result", ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)"])

# Create a DataFrame for input features
input_features = np.array([[age, int(sex.split()[1]), int(chest_pain.split()[0]), bp, cholesterol,
                            int(fbs.split()[0]), int(ekg.split()[0]), max_hr,
                            int(exercise_angina.split()[0]), st_depression, int(slope.split()[0]),
                            vessels_fluro, int(thallium.split()[0])]])

# Button for prediction
if st.button("Predict"):
    # Predict the result
    prediction = model.predict(input_features)

    # Convert prediction output to a human-readable form
    result = 'Presence of heart disease' if prediction[0] == 1 else 'Absence of heart disease'

    # Display the result
    st.success(result)

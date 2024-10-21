from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained logistic regression model
with open('D:/Users/user/Desktop/Heart Disease Prediction/model2.pkl', 'rb') as file:
    model = pickle.load(file)
# model = pickle.load(open('D:/Users/user/Desktop/Heart Disease Prediction/model2.pkl', 'rb'))

# Home route to serve the HTML form
@app.route('/')
def home():
    return render_template('heart disease1.html')  # Connect to your form page here

# Prediction route for handling form data
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = request.form

    # Convert form values to appropriate types (ensure these match the order expected by the model)
    input_features = [int(data['age']),
                      int(data['sex']),
                      int(data['chest_pain']),
                      int(data['bp']),
                      int(data['cholesterol']),
                      int(data['fbs']),
                      int(data['ekg']),
                      int(data['max_hr']),
                      int(data['exercise_angina']),
                      float(data['st_depression']),
                      int(data['slope']),
                      int(data['vessels_fluro']),
                      int(data['thallium'])]

    # Convert to numpy array and reshape for model input
    input_array = np.array([input_features])

    # Predict the result
    prediction = model.predict(input_array)

    # Convert prediction output to a human-readable form
    result = 'Presence' if prediction[0] == 1 else 'Absence'

    # Render the result back on the page
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

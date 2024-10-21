
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

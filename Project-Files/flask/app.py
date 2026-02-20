from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = './payments.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Make sure payments.pkl is in the flask directory.")
    exit()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve form data
            step = int(request.form['step'])
            transaction_type = int(request.form['type']) # Already encoded
            amount = float(request.form['amount'])
            oldbalanceOrg = float(request.form['oldbalanceOrg'])
            newbalanceOrig = float(request.form['newbalanceOrig'])
            oldbalanceDest = float(request.form['oldbalanceDest'])
            newbalanceDest = float(request.form['newbalanceDest'])
            isFlaggedFraud = int(request.form['isFlaggedFraud'])

            # Create a DataFrame for prediction
            # The order of columns must match the training data:
            # ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
            input_data = pd.DataFrame([[step, transaction_type, amount, oldbalanceOrg,
                                        newbalanceOrig, oldbalanceDest, newbalanceDest,
                                        isFlaggedFraud]],
                                      columns=['step', 'type', 'amount', 'oldbalanceOrg',
                                               'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                                               'isFlaggedFraud'])

            # Make prediction
            prediction = model.predict(input_data)[0]

            return render_template('submit.html', prediction=prediction)

        except Exception as e:
            return render_template('submit.html', prediction=-1, error=str(e)) # -1 for error state, or handle differently
    return render_template('predict.html')

@app.route('/submit')
def submit():
    # This route is mainly for displaying results from a POST request to /predict
    # Direct access might show an empty page or an error if no prediction is passed.
    return "Please submit your prediction request through the /predict page."

if __name__ == '__main__':
    app.run(debug=True)

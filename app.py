from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and encoders
model = joblib.load("churn_model.pkl")
encoders = joblib.load("encoders.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    # Get data from the form
    age = int(request.form["age"])
    gender = encoders['Gender'].transform([request.form["gender"]])[0]
    tenure = int(request.form["tenure"])
    monthly = float(request.form["monthly"])
    contract = encoders['ContractType'].transform([request.form["contract"]])[0]
    internet = encoders['InternetService'].transform([request.form["internet"]])[0]
    total = float(request.form["total"])
    support = encoders['TechSupport'].transform([request.form["support"]])[0]

    # Prepare input for model
    input_data = np.array([[age, gender, tenure, monthly, contract, internet, total, support]])
    prediction = model.predict(input_data)[0]

    result = "Yes" if prediction == 1 else "No"
    return render_template("index.html", prediction_text=f"Churn Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)

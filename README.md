# Customer Churn Prediction — Web App

A machine learning web application that predicts whether a customer is likely to churn (leave the service) based on their profile and account details. Built with a scikit-learn RandomForest classifier and served through a Flask web interface.

## What it does

Users enter a customer's details — age, gender, tenure, monthly charges, contract type, internet service, total charges, and tech support status — through a simple web form. The app runs this input through a trained ML model and instantly returns a churn prediction (Yes/No).

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn (RandomForestClassifier), pandas, joblib
- **Frontend:** HTML, CSS (Jinja2 templating via Flask)
- **Deployment:** Render (Gunicorn WSGI server)

## Project Structure

```
├── app.py                 # Flask application (routes + prediction logic)
├── train_model.py         # Script to train the RandomForest model from scratch
├── cleaned_dataset.csv    # Preprocessed customer dataset (1,000 records)
├── churn_model.pkl        # Trained model (saved with joblib)
├── encoders.pkl           # Saved LabelEncoders for categorical fields
├── templates/
│   └── index.html         # Form UI + prediction result display
├── static/
│   └── background.jpg.jpg # Background image for the UI
├── requirements.txt        # Python dependencies
└── Procfile                # Start command for deployment (Render/Heroku-style)
```

## How the model was built

1. Loaded and cleaned a customer dataset with 8 features (age, gender, tenure, monthly charges, contract type, internet service, total charges, tech support) and a target label (`Churn`).
2. Encoded categorical fields (`Gender`, `ContractType`, `InternetService`, `TechSupport`, `Churn`) using `LabelEncoder`.
3. Split the data into training and test sets (80/20).
4. Trained a `RandomForestClassifier` (100 estimators, max depth 5) on the training set.
5. Evaluated performance using precision, recall, and F1-score on the held-out test set.
6. Saved the trained model and encoders with `joblib` for reuse in the Flask app.

## Running it locally

1. Clone the repo:
   ```
   git clone https://github.com/Pritihambarde/Customer-Loss-Prediction.git
   cd Customer-Loss-Prediction
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

## Retraining the model

If you want to retrain on updated data:
```
python train_model.py
```
This regenerates `churn_model.pkl` and `encoders.pkl` from `cleaned_dataset.csv`.

## Notes

- The current dataset is a small (1,000-record) sample used for learning purposes; performance on this dataset is not representative of real-world production data, which would require a larger, more diverse dataset for a robust evaluation.
- Future improvements: add input validation, try additional models (Logistic Regression, Gradient Boosting) for comparison, and expand the dataset.

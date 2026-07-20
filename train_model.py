import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Step 1: Load dataset
df = pd.read_csv("cleaned_dataset.csv")

# Step 2: Encode categorical columns with separate encoders
categorical_cols = ['Gender', 'ContractType', 'InternetService', 'TechSupport', 'Churn']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save encoder for later use

# Step 3: Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the mod
# Step 5: Train the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))

# Step 7: Save model and encoders
joblib.dump(model, "churn_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("\n✅ Model and encoders saved successfully!")


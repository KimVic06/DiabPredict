import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
df = pd.read_csv(url, names=columns)

print("Shape:", df.shape)
print(df.head())
print(df['Outcome'].value_counts())

# Replace 0 with NaN for columns where 0 is a physiologically invalid value
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split BEFORE imputing, to avoid leaking test-set info into the median
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Impute missing values using medians learned ONLY from the training set
imputer = SimpleImputer(strategy='median')
X_train[cols_with_zeros] = imputer.fit_transform(X_train[cols_with_zeros])
X_test[cols_with_zeros] = imputer.transform(X_test[cols_with_zeros])

print("Missing values in training set after imputation:")
print(X_train.isnull().sum())
print("Missing values in test set after imputation:")
print(X_test.isnull().sum())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200, random_state=42, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Save model, scaler, and imputer (all three are needed at inference time)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)

print("Model, scaler, and imputer saved successfully.")
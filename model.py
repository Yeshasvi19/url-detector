import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np
import tempfile
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Verify the file path
file_path = r'D:\URL\url_dataset.csv'  # Use raw string

if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at path {file_path} does not exist. Please check the path and try again.")

# Load the dataset
data = pd.read_csv(file_path)
print(data.head())

def extract_features(url, domain_info):
    features = {}
    
    # URL length
    features['url_length'] = len(url)
    
    # Count special characters
    features['special_chars'] = sum([1 for char in url if char in ['/', '?', '&', '=', '-', '_', '%', '.', ':']])
    
    # Count digits
    features['digit_count'] = sum([1 for char in url if char.isdigit()])
    
    # Count letters
    features['letter_count'] = sum([1 for char in url if char.isalpha()])
    
    # Check if the URL has an IP address
    features['has_ip'] = 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url) else 0
    
    # Domain-based features
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    features['domain_length'] = len(domain)
    features['subdomain_count'] = domain.count('.')
    
    # Domain days age
    domain_creation_date = domain_info.get(domain, {}).get('creation_date')
    if domain_creation_date:
        domain_days_age = (datetime.now() - domain_creation_date).days
    else:
        domain_days_age = -1  # Unknown age
    
    features['domain_days_age'] = domain_days_age
    
    # Is registered
    features['is_registered'] = 1 if domain_info.get(domain, {}).get('is_registered', False) else 0
    
    return features

# Mock domain information
# Replace with actual data retrieval for production use
domain_info = {
    'example.com': {'creation_date': datetime(2020, 1, 1), 'is_registered': True},
    'another-example.com': {'creation_date': datetime(2015, 6, 15), 'is_registered': True},
    # Add more domain info as needed
}

# Apply feature extraction
data_features = data['url'].apply(lambda url: extract_features(url, domain_info)).apply(pd.Series)
print(data_features.head())

# Combine features with the original data
data = pd.concat([data, data_features], axis=1)

# Map 'type' to numerical values
type_mapping = {'benign': 0, 'phishing': 1, 'defacement': 2}
data['type'] = data['type'].map(type_mapping)

# Drop the original URL column
data.drop(columns=['url'], inplace=True)

# Check for and handle NaN values in features
if data.isnull().values.any():
    print("NaN values detected in features. Replacing NaN values with column means.")
    data.fillna(data.mean(), inplace=True)

# Check for and handle infinite values in features
if np.isinf(data.values).any():
    print("Infinite values detected in features. Replacing infinite values with large finite numbers.")
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)

# Ensure all values are within a valid range
print("Checking for overly large values.")
for column in data.columns:
    if data[column].dtype in [np.float64, np.int64]:
        data[column] = np.clip(data[column], -1e12, 1e12)  # Increased range to handle larger values

# Separate features and labels
X = data.drop(columns=['type'])
y = data['type']

# Ensure y is of integer type
y = y.astype(int)

# Check for and handle NaN values in target variable
if y.isnull().values.any():
    print("NaN values detected in target variable. Removing rows with NaN target values.")
    X = X[~y.isnull()]
    y = y.dropna()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified split of the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Initialize the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Train the model using cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='accuracy')
print("Cross-validation Accuracy Scores:", cross_val_scores)
print("Mean Cross-validation Accuracy:", cross_val_scores.mean())

# Train the model on the full training set
xgb_model.fit(X_train, y_train)

# Make predictions
xgb_y_pred = xgb_model.predict(X_test)
xgb_y_proba = xgb_model.predict_proba(X_test)

# Evaluate the model
xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
xgb_roc_auc = roc_auc_score(y_test, xgb_y_proba, multi_class='ovr')
xgb_report = classification_report(y_test, xgb_y_pred)

print("XGBoost Accuracy:", xgb_accuracy)
print("XGBoost ROC AUC Score:", xgb_roc_auc)
print("XGBoost Classification Report:\n", xgb_report)

# Create the output directory if it doesn't exist
output_dir = r'D:\URL'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model and the scaler to specific paths
output_model_path = os.path.join(output_dir, 'malicious_url_detector.pkl')
output_scaler_path = os.path.join(output_dir, 'scaler.pkl')

try:
    joblib.dump(xgb_model, output_model_path)
    print(f"Model saved successfully at {output_model_path}")
    joblib.dump(scaler, output_scaler_path)
    print(f"Scaler saved successfully at {output_scaler_path}")
except PermissionError as e:
    print(f"PermissionError: {e}. Attempting to save to a temporary directory.")
    temp_dir = tempfile.gettempdir()
    output_model_path = os.path.join(temp_dir, 'malicious_url_detector.pkl')
    joblib.dump(xgb_model, output_model_path)
    print(f"Model saved successfully at {output_model_path}")
    output_scaler_path = os.path.join(temp_dir, 'scaler.pkl')
    joblib.dump(scaler, output_scaler_path)
    print(f"Scaler saved successfully at {output_scaler_path}")

# Save the dataset with predictions to a CSV file
output_csv_path = os.path.join(output_dir, 'url_dataset_with_predictions.csv')

# Map predictions back to their original names
inverse_type_mapping = {v: k for k, v in type_mapping.items()}
X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df['predicted_type'] = xgb_y_pred
X_test_df['predicted_type'] = X_test_df['predicted_type'].map(inverse_type_mapping)

X_test_df.to_csv(output_csv_path, index=False)
print(f"Dataset with predictions saved successfully at {output_csv_path}")

# Save the features (X) and labels (y) data to separate CSV files
X_train_path = os.path.join(output_dir, 'X_train.csv')
y_train_path = os.path.join(output_dir, 'y_train.csv')
X_test_path = os.path.join(output_dir, 'X_test.csv')
y_test_path = os.path.join(output_dir, 'y_test.csv')

X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_train_df.to_csv(X_train_path, index=False)
y_train.to_csv(y_train_path, index=False)
X_test_df.to_csv(X_test_path, index=False)
y_test.to_csv(y_test_path, index=False)

print(f"X_train saved to {X_train_path}")
print(f"y_train saved to {y_train_path}")
print(f"X_test saved to {X_test_path}")
print(f"y_test saved to {y_test_path}")

# Load the model
loaded_model = joblib.load(output_model_path)
print("Model loaded successfully")

def classify_url(url, domain_info, model, scaler):
    # Extract features
    features = extract_features(url, domain_info)
    features_df = pd.DataFrame([features])
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    return prediction

# User input URL
user_url = input("Enter a URL to classify: ")
prediction = classify_url(user_url, domain_info, loaded_model, scaler)
print(f"The URL '{user_url}' is classified as: {inverse_type_mapping[prediction]}")

# Plotting functions
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importance)), importance[indices], align="center")
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importance)])
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Plot feature importance
plot_feature_importance(loaded_model, X.columns)

# Plot confusion matrix
plot_confusion_matrix(y_test, xgb_y_pred, [inverse_type_mapping[i] for i in range(len(type_mapping))])
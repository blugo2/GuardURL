from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch UCI dataset
phishing_websites = fetch_ucirepo(id=327)

# Data and target
X = phishing_websites.data.features
y = phishing_websites.data.targets

# Features available in the UCI dataset
uci_features = [
    'having_ip_address', 'url_length', 'shortining_service',
    'having_at_symbol', 'double_slash_redirecting', 'prefix_suffix',
    'sslfinal_state', 'domain_registration_length', 'favicon', 'port',
    'https_token'
]

# Extract only the features we will use for compatibility
X_uci = X[uci_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_uci, y, test_size=0.3, random_state=42)

# Train a Random Forest model on the common features
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions and evaluation on UCI dataset
y_pred = clf.predict(X_test)
print("Evaluation on UCI Dataset:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

# Load custom dataset (balanced_urls.csv)
file_path = 'balanced_urls.csv'
balanced_data = pd.read_csv(file_path)

# Extract and map features for the new dataset
balanced_data['having_ip_address'] = balanced_data['url'].str.contains(r'\d+\.\d+\.\d+\.\d+').astype(int)
balanced_data['url_length'] = balanced_data['url'].apply(len)
balanced_data['having_at_symbol'] = balanced_data['url'].str.contains('@').astype(int)
balanced_data['prefix_suffix'] = balanced_data['url'].str.contains('-').astype(int)
balanced_data['https_token'] = balanced_data['url'].str.startswith('https').astype(int)

# Placeholder values for features that cannot be derived
balanced_data['shortining_service'] = 0
balanced_data['double_slash_redirecting'] = 0
balanced_data['sslfinal_state'] = 1
balanced_data['domain_registration_length'] = 1
balanced_data['favicon'] = 1
balanced_data['port'] = 0

# Ensure only common features are used
X_balanced = balanced_data[uci_features]
X_balanced = X_balanced.fillna(0)
y_balanced = balanced_data['result']

# Make predictions on the balanced dataset
balanced_predictions = clf.predict(X_balanced)
balanced_probabilities = clf.predict_proba(X_balanced)[:, 1]

# Evaluate performance on the balanced dataset
print("\nEvaluation on Balanced Dataset:")
print(confusion_matrix(y_balanced, balanced_predictions))
print(classification_report(y_balanced, balanced_predictions))
print("ROC-AUC Score:", roc_auc_score(y_balanced, balanced_probabilities))

# Visualizations
class_names = ["Malicious", "Benign", "Potentially Malicious"]

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=feature_names)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()

def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

def plot_class_distribution(labels, title):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=[class_names[u] for u in unique], y=counts)
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Counts")
    plt.show()

# Plot feature importance
plot_feature_importance(clf, uci_features)

# Confusion matrices
plot_confusion_matrix(confusion_matrix(y_test, y_pred), class_names, "Confusion Matrix (UCI Dataset)")
plot_confusion_matrix(confusion_matrix(y_balanced, balanced_predictions), class_names, "Confusion Matrix (Balanced Dataset)")

# Plot ROC curve for UCI dataset
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(clf, X_test, y_test, name="UCI Dataset")
plt.title("ROC Curve (UCI Dataset)")
plt.show()

# Plot ROC curve for Balanced dataset
try:
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_balanced, balanced_probabilities, name="Balanced Dataset")
    plt.title("ROC Curve (Balanced Dataset)")
    plt.show()
except ValueError as e:
    print(f"Error in plotting ROC Curve for Balanced Dataset: {e}")

# Plot class distributions
plot_class_distribution(y_test, "Class Distribution (UCI Dataset)")
plot_class_distribution(y_balanced, "Class Distribution (Balanced Dataset)")



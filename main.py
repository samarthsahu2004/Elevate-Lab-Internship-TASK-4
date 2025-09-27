# ==============================================================================
# BREAST CANCER CLASSIFICATION WITH LOGISTIC REGRESSION
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. SETUP AND DATA LOADING
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import kagglehub

# Set plot style
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

# Download and load the dataset
print("Downloading the dataset...")
path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
file_path = f"{path}/data.csv"
df = pd.read_csv(file_path)
print("Dataset loaded successfully!")

# ------------------------------------------------------------------------------
# 2. DATA PREPROCESSING
# ------------------------------------------------------------------------------
# Drop unnecessary columns
df = df.drop(['id', 'Unnamed: 32'], axis=1)

# Encode the target variable 'diagnosis' (M=1, B=0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

print("\n--- Data after Preprocessing ---")
print(df.head())

# ------------------------------------------------------------------------------
# 3. MODEL BUILDING AND TRAINING
# ------------------------------------------------------------------------------
# Define features (X) and target (y)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print("\nLogistic Regression model trained successfully!\n")

# ------------------------------------------------------------------------------
# 4. MODEL EVALUATION
# ------------------------------------------------------------------------------
# Make predictions
y_pred = model.predict(X_test_scaled)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign (0)', 'Malignant (1)'],
            yticklabels=['Benign (0)', 'Malignant (1)'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# --- Classification Report ---
print("--- Classification Report (Threshold = 0.5) ---")
print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malignant (1)']))

# --- ROC-AUC Curve ---
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
# 5. THRESHOLD TUNING EXAMPLE
# ------------------------------------------------------------------------------
print("\n--- Threshold Tuning Example ---")
# Get new predictions with a lower threshold of 0.4
y_pred_new_threshold = (model.predict_proba(X_test_scaled)[:, 1] >= 0.4).astype(int)

print("Classification Report with Threshold = 0.4:")
print(classification_report(y_test, y_pred_new_threshold, target_names=['Benign (0)', 'Malignant (1)']))

print("\nAnalysis complete.")

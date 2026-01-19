"""
Simple training script for HAR project
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os

# Create folders if they don't exist
if not os.path.exists('notebook/models'):
    os.makedirs('notebook/models')
if not os.path.exists('notebook/performance'):
    os.makedirs('notebook/performance')

print("Loading data...")

# Load features
features = pd.read_csv('data/UCI HAR Dataset/features.txt', sep=r'\s+', header=None)
feature_names = features[1].tolist()

# Load activities
activities = pd.read_csv('data/UCI HAR Dataset/activity_labels.txt', sep=r'\s+', header=None)
activity_dict = dict(zip(activities[0], activities[1]))

# Load training data
X_train = pd.read_csv('data/UCI HAR Dataset/train/X_train.txt', sep=r'\s+', header=None)
X_train.columns = feature_names
y_train = pd.read_csv('data/UCI HAR Dataset/train/y_train.txt', header=None)
y_train = y_train[0].values - 1  # Convert to 0-indexed

# Load test data
X_test = pd.read_csv('data/UCI HAR Dataset/test/X_test.txt', sep=r'\s+', header=None)
X_test.columns = feature_names
y_test = pd.read_csv('data/UCI HAR Dataset/test/y_test.txt', header=None)
y_test = y_test[0].values - 1  # Convert to 0-indexed

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

print("\nTraining model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    tree_method='hist'
)
# Convert to numpy arrays for compatibility with newer XGBoost
model.fit(X_train.values, y_train)

print("\nEvaluating...")
predictions = model.predict(X_test.values)
accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nDetailed results:")
activity_names = list(activity_dict.values())
print(classification_report(y_test, predictions, target_names=activity_names))

# Save model
print("\nSaving model...")
joblib.dump(model, 'notebook/models/model.pkl')
joblib.dump(feature_names, 'notebook/models/features.pkl')

# Save performance
results = pd.DataFrame({
    'accuracy': [accuracy]
})
results.to_csv('notebook/performance/metrics.csv', index=False)

print("Done!")

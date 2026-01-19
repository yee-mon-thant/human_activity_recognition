"""
Test script to validate XGBoost model performance
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("=" * 60)
print("XGBoost Model Performance Test")
print("=" * 60)

# Load model
print("\n1. Loading model...")
model = joblib.load('../../notebook/models/xgb_model.pkl')
feature_names = joblib.load('../../notebook/models/features.pkl')
print(f"✓ Model loaded: {type(model).__name__}")
print(f"✓ Features loaded: {len(feature_names)} features")

# Load test data
print("\n2. Loading test data...")
X_test = pd.read_csv('../../data/UCI HAR Dataset/test/X_test.txt', sep=r'\s+', header=None)
X_test.columns = feature_names
y_test = pd.read_csv('../../data/UCI HAR Dataset/test/y_test.txt', header=None)
y_test = y_test[0].values - 1  # Convert to 0-indexed

print(f"✓ Test samples: {len(X_test)}")
print(f"✓ Test features: {X_test.shape[1]}")

# Activity names
activities = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'
}

# Make predictions
print("\n3. Making predictions...")
y_pred = model.predict(X_test.values)
y_proba = model.predict_proba(X_test.values)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'=' * 60}")
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'=' * 60}")

# Detailed classification report
print("\n4. Classification Report:")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=list(activities.values())))

# Confusion matrix
print("\n5. Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Per-class accuracy
print("\n6. Per-Class Performance:")
print("-" * 60)
for i, activity in activities.items():
    class_mask = y_test == i
    if class_mask.sum() > 0:
        class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
        class_samples = class_mask.sum()
        print(f"{activity:20s}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_samples} samples")

# Average confidence
print("\n7. Prediction Confidence:")
print("-" * 60)
avg_confidence = np.mean([y_proba[i][pred] for i, pred in enumerate(y_pred)])
print(f"Average confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")

# Confidence per class
print("\nConfidence by predicted class:")
for i, activity in activities.items():
    class_mask = y_pred == i
    if class_mask.sum() > 0:
        class_conf = np.mean([y_proba[j][i] for j in np.where(class_mask)[0]])
        print(f"  {activity:20s}: {class_conf:.4f}")

# Model parameters
print("\n8. Model Configuration:")
print("-" * 60)
print(f"Number of estimators: {model.n_estimators}")
print(f"Max depth: {model.max_depth}")
print(f"Learning rate: {model.learning_rate}")

# Feature importance (top 10)
print("\n9. Top 10 Most Important Features:")
print("-" * 60)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
for i, idx in enumerate(indices, 1):
    print(f"{i:2d}. {feature_names[idx]:30s}: {importances[idx]:.4f}")

# Save confusion matrix plot
print("\n10. Saving visualizations...")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(activities.values()),
            yticklabels=list(activities.values()))
plt.title(f'XGBoost Model - Confusion Matrix\nAccuracy: {accuracy:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")

# Feature importance plot
plt.figure(figsize=(10, 6))
top_n = 15
indices = np.argsort(importances)[::-1][:top_n]
plt.barh(range(top_n), importances[indices])
plt.yticks(range(top_n), [feature_names[i] for i in indices])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("✓ Saved: feature_importance.png")

print("\n" + "=" * 60)
print("Testing Complete!")
print("=" * 60)

# Summary
print("\nSUMMARY:")
print(f"  Model: XGBoost Classifier")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Confidence: {avg_confidence:.4f}")
print(f"  Test samples: {len(X_test)}")
print(f"  Classes: {len(activities)}")

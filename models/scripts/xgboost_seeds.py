# %%
"""This notebook is to train XGBoost on wheat seeds prediction dataset"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# %%
data = pd.read_csv('../datasets/Seed_Data.csv')
print("Data shape:", data.shape)
data.head()

# %%
print("Column names:", data.columns.tolist())
print("Original data shape:", data.shape)

# %%
data = data.dropna()
print("Data shape after removing NaN:", data.shape)


# %%
print("Missing values per column:", data.isnull().sum())

# %%
X = data.drop(columns=["target"])
y = data["target"]

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("Target classes:", sorted(y.unique()))

# %%
print("\nFeature data types:")
print(X.dtypes)
print("\nTarget data type:", y.dtype)


# %%
print("\nFeature statistics:")
print(X.describe())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Training target distribution:", y_train.value_counts().sort_index())
print("Test target distribution:", y_test.value_counts().sort_index())

# %%
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    learning_rate=0.1,
    n_estimators=10000,        # Big upper bound (acts as max)
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    early_stopping_rounds=50,
    verbosity=1
)

# %%
xgb_model.fit(X_train, y_train,
eval_set = [(X_train, y_train), (X_test, y_test)],
verbose=True
)
print("✅ XGBoost model training completed!")

# %%
print("Making predictions...")
y_pred = xgb_model.predict(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# %%
print("Generating feature importance plot...")
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

# %%
k = min(20, len(importances))

plt.figure(figsize=(12, 8))
plt.title("XGBoost Feature Importances - Wheat Seeds Dataset", fontsize=16, fontweight='bold')
plt.bar(range(k), importances[indices[:k]], align="center", alpha=0.7, color='lightcoral')
plt.xticks(range(k), X.columns[indices[:k]], rotation=45, ha='right')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.tight_layout()
plt.show()

# %%
print("\nFeature Importance Ranking:")
for i in range(len(importances)):
    feature_idx = indices[i]
    print(f"{i+1:2d}. {X.columns[feature_idx]:<20} {importances[feature_idx]:.4f}")


# %%
print("\nSample Predictions:")
sample = X_test.head()
sample_predictions = xgb_model.predict(sample)
sample_probabilities = xgb_model.predict_proba(sample)

print("Predictions:", sample_predictions.tolist())
print("Actual:", y_test.head().tolist())

# %%
print("\nPrediction Probabilities:")
class_labels = sorted(y.unique())
for i, (pred, prob) in enumerate(zip(sample_predictions, sample_probabilities)):
    print(f"Sample {i+1}: Predicted class {pred}")
    for j, class_label in enumerate(class_labels):
        print(f"  Class {class_label}: {prob[j]:.3f}")
    print()

# Save everything in a single file
model_package = {
    'model': xgb_model,
    'feature_columns': X.columns.tolist(),
    'target_classes': sorted(y.unique()),
    'model_type': 'XGBoost',
    'accuracy': accuracy,
    'dataset': 'Wheat Seeds',
    'feature_importances': dict(zip(X.columns, importances))
}

# %%
model_filename = "xgboost_wheat_seeds_complete.pkl"
joblib.dump(model_package, model_filename)
print(f"✅ Complete model package saved as {model_filename}")
print("Package contains: model, feature_columns, target_classes, model_type, accuracy, dataset, feature_importances")


# %%
print("\n" + "="*60)
print("XGBoost Wheat Seeds Classification - Training Summary")
print("="*60)
print(f"Final Accuracy: {accuracy:.4f}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of classes: {len(class_labels)}")
print(f"Classes: {class_labels}")
print(f"Model package saved: {model_filename}")
print("="*60)

# %%




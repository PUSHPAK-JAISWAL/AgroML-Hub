# %%
"""This is a notebook to train XGBoost on forest prediction dataset"""

# %%
import pandas as pd

# %%
data = pd.read_csv('../datasets/forest.csv')

# %%
data.head()

# %%
data.columns

# %%
import numpy as np

# %%
wilderness_cols = [c for c in data.columns if c.startswith("Wilderness_Area")]
data["Wilderness_Area"] = np.argmax(data[wilderness_cols].values, axis=1) + 1
data = data.drop(columns=wilderness_cols)

# %%
soil_cols = [c for c in data.columns if c.startswith("Soil_Type")]
data["Soil_Type"] = np.argmax(data[soil_cols].values, axis=1) + 1
data = data.drop(columns=soil_cols)

# %%
data = data.drop(columns=["Id"])

print(data.head())
print("\nColumns:", data.columns.tolist())


# %%
data.head()

# %%
data = data.dropna()

# %%
print(data.shape)

# %%
X = data.drop(columns=["Cover_Type"])
y = data["Cover_Type"]
y = y-1

print(X.shape, y.shape)
print("Unique target values:", sorted(y.unique()))


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=300,        # More trees than RF default
    max_depth=6,             # Controlled depth to prevent overfitting
    learning_rate=0.1,       # Standard learning rate
    subsample=0.8,           # Row sampling to prevent overfitting
    colsample_bytree=0.8,    # Column sampling
    random_state=42,
    n_jobs=-1,               # Use all CPU cores
    eval_metric='mlogloss',  # For multiclass classification
    early_stopping_rounds=50 # Stop if no improvement for 50 rounds
)

# %%
eval_set = [(X_train, y_train), (X_test, y_test)]

# %%
xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=True  # Set to True to see training progress
)

# %%
from sklearn.metrics import accuracy_score, classification_report

y_pred = xgb_model.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Take top k features
k = min(20, len(importances))

plt.figure(figsize=(12,6))
plt.title("XGBoost Feature Importances")
plt.bar(range(k), importances[indices[:k]], align="center")
plt.xticks(range(k), X.columns[indices[:k]], rotation=90)
plt.tight_layout()
plt.show()

# %%
sample = X_test.head()

# Predict on them
predictions = xgb_model.predict(sample)

print("XGBoost Predictions:", predictions.tolist())
print("Actual:", y_test.head().tolist())

# %%
import joblib

# Save trained XGBoost model
joblib.dump(xgb_model, "xgboost_forest_model.pkl")
print("✅ XGBoost model saved as xgboost_forest_model.pkl")

# Optional: Print training history
print(f"\nBest iteration: {xgb_model.best_iteration}")
print(f"Best score: {xgb_model.best_score:.4f}")

# %%




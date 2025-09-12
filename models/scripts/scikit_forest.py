# %%
"""This is a note book to train random forest on forest prediction datatset"""

# %%
import pandas as pd

# %%
data = pd.read_csv(r'../datasets/forest.csv')

# %%
data.head()

# %%
data.columns

# %%
import numpy as np

# --- Collapse Wilderness Areas ---
wilderness_cols = [c for c in data.columns if c.startswith("Wilderness_Area")]
data["Wilderness_Area"] = np.argmax(data[wilderness_cols].values, axis=1) + 1
data = data.drop(columns=wilderness_cols)

# --- Collapse Soil Types ---
soil_cols = [c for c in data.columns if c.startswith("Soil_Type")]
data["Soil_Type"] = np.argmax(data[soil_cols].values, axis=1) + 1
data = data.drop(columns=soil_cols)

# Drop Id (not useful for ML)
data = data.drop(columns=["Id"])

print(data.head())
print("\nColumns:", data.columns.tolist())


# %%
data.head()

# %%
# Drop rows with any null values
data = data.dropna()

print(data.shape)


# %%

# Features and Target
X = data.drop(columns=["Cover_Type"])
y = data["Cover_Type"]

print(X.shape, y.shape)


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# %%
from sklearn.ensemble import RandomForestClassifier

# Initialize model
rf = RandomForestClassifier(  # number of trees
    max_depth=None,   # let trees grow fully
    random_state=42,
    n_jobs=-1         # use all CPU cores
)

# Train
rf.fit(X_train, y_train)


# %%
from sklearn.metrics import accuracy_score, classification_report

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# %%
import matplotlib.pyplot as plt
import numpy as np

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Take top k features (max = number of features)
k = min(20, len(importances))

plt.figure(figsize=(12,6))
plt.title("Feature Importances")
plt.bar(range(k), importances[indices[:k]], align="center")
plt.xticks(range(k), X.columns[indices[:k]], rotation=90)
plt.show()


# %%
# Take first 5 rows from X_test
sample = X_test.head()

# Predict on them
predictions = rf.predict(sample)

print("Predictions:", predictions.tolist())
print("Actual:", y_test.head().tolist())


# %%
import joblib

# Save trained model
joblib.dump(rf, "random_forest_model.pkl")
print("✅ Model saved as random_forest_model.pkl")

# %%




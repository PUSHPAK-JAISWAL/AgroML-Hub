# %%
"""This notebook is to train random forest on wheat seeds prediction dataset"""

# %%
import pandas as pd

# %%
data = pd.read_csv('../datasets/Seed_Data.csv')

# %%
data.head()

# %%
data.columns

# %%
data.shape

# %%
data = data.dropna()

print(data.shape)

# %%
X = data.drop(columns=["target"])

y = data["target"]

print(X.shape,y.shape)

# %%
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size = 0.2,random_state=42,stratify = y
)

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    max_depth = None,
    random_state = 42,
)

# %%
rf.fit(X_train,y_train)

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
joblib.dump(rf, "random_Seeds_model.pkl")
print("✅ Model saved as random_Seeds_model.pkl")

# %%




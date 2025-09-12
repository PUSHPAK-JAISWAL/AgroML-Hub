# %%
"""This is a note book to train random forest prediction datasets for mushrooms dataset"""

# %%
import pandas as pd

# %%
data = pd.read_csv('../datasets/mushrooms.csv')

# %%
data.head()

# %%
mappings = {
    "class":{
        "p":"poisonous",
        "e":"edible",
    },
    "cap-shape": {
        "b": "bell", "c": "conical", "x": "convex",
        "f": "flat", "k": "knobbed", "s": "sunken"
    },
    "cap-surface": {
        "f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"
    },
    "cap-color": {
        "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green",
        "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"
    },
    "bruises": {"t": "bruises", "f": "no"},
    "odor": {
        "a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul",
        "m": "musty", "n": "none", "p": "pungent", "s": "spicy"
    },
    "gill-attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill-spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill-size": {"b": "broad", "n": "narrow"},
    "gill-color": {
        "k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray",
        "r": "green", "o": "orange", "p": "pink", "u": "purple", "e": "red",
        "w": "white", "y": "yellow"
    },
    "stalk-shape": {"e": "enlarging", "t": "tapering"},
    "stalk-root": {
        "b": "bulbous", "c": "club", "u": "cup", "e": "equal",
        "z": "rhizomorphs", "r": "rooted", "?": "missing"
    },
    "stalk-surface-above-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-surface-below-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-color-above-ring": {
        "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange",
        "p": "pink", "e": "red", "w": "white", "y": "yellow"
    },
    "stalk-color-below-ring": {
        "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange",
        "p": "pink", "e": "red", "w": "white", "y": "yellow"
    },
    "veil-type": {"p": "partial", "u": "universal"},
    "veil-color": {"n": "brown", "o": "orange", "w": "white", "y": "yellow"},
    "ring-number": {"n": "none", "o": "one", "t": "two"},
    "ring-type": {
        "c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large",
        "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"
    },
    "spore-print-color": {
        "k": "black", "n": "brown", "b": "buff", "h": "chocolate",
        "r": "green", "o": "orange", "u": "purple", "w": "white", "y": "yellow"
    },
    "population": {
        "a": "abundant", "c": "clustered", "n": "numerous",
        "s": "scattered", "v": "several", "y": "solitary"
    },
    "habitat": {
        "g": "grasses", "l": "leaves", "m": "meadows", "p": "paths",
        "u": "urban", "w": "waste", "d": "woods"
    }
}

# %%
data = data.replace(mappings)

# %%
data.head()

# %%
data.columns

# %%
data.columns.isnull().sum()

# %%
X = data.drop(columns=["class"])
y = data[["class"]]

print(X.shape,y.shape)

# %%
from sklearn.preprocessing import LabelEncoder

# Encode features
X_encoded = X.apply(LabelEncoder().fit_transform)

# Encode target
y_encoded = LabelEncoder().fit_transform(y)


X_encoded.head()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    max_depth=None,
    random_state= 42
)

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
print("Actual:", y_test[:5].tolist())   # use slicing for numpy arrays

# %%
import joblib

# Save trained model
joblib.dump(rf, "random_mushrooms_model.pkl")
print("✅ Model saved as random_mushrooms_model.pkl")

# %%




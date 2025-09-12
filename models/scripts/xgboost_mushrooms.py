# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb


# %%
data = pd.read_csv('../datasets/mushrooms.csv')
print("Data shape:", data.shape)
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
print("Data after mapping:")
data.head()

# %%
print("Column names:", data.columns.tolist())
print("Null values per column:", data.isnull().sum())

# %%
X = data.drop(columns=["class"])
y = data[["class"]]

# %%
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

# %%
print("Encoding categorical features...")
X_encoded = X.apply(LabelEncoder().fit_transform)

# %%
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y.values.ravel())

print("Encoded feature matrix shape:", X_encoded.shape)
print("Encoded target shape:", y_encoded.shape)
print("Target classes:", label_encoder_y.classes_)

# %%
X_encoded.head()

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# %%
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# %%
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # For binary classification                 # Maximum tree depth
    learning_rate=0.1,            # Learning rate
    n_estimators=10000,             # Number of boosting rounds
    subsample=0.8,                # Subsample ratio of training instances
    colsample_bytree=0.8,         # Subsample ratio of features
    random_state=42,              # For reproducibility
    eval_metric='logloss',
    early_stopping_rounds = 50,        # Evaluation metric
    verbosity=1                   # Control verbosity
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
print(classification_report(y_test, y_pred, target_names=label_encoder_y.classes_))

# %%
print("Generating feature importance plot...")
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

# %%
k = min(20, len(importances))

plt.figure(figsize=(12, 8))
plt.title("XGBoost Feature Importances", fontsize=16, fontweight='bold')
plt.bar(range(k), importances[indices[:k]], align="center", alpha=0.7, color='skyblue')
plt.xticks(range(k), X.columns[indices[:k]], rotation=45, ha='right')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.tight_layout()
plt.show()

# %%
print("\nTop 10 Most Important Features:")
for i in range(min(10, len(importances))):
    feature_idx = indices[i]
    print(f"{i+1:2d}. {X.columns[feature_idx]:<25} {importances[feature_idx]:.4f}")


# %%
print("\nSample Predictions:")
sample = X_test.head()
sample_predictions = xgb_model.predict(sample)
sample_probabilities = xgb_model.predict_proba(sample)

print("Predictions:", [label_encoder_y.inverse_transform([pred])[0] for pred in sample_predictions])
print("Actual:", [label_encoder_y.inverse_transform([actual])[0] for actual in y_test[:5]])
print("\nPrediction Probabilities:")
for i, (pred, prob) in enumerate(zip(sample_predictions, sample_probabilities)):
    class_name = label_encoder_y.inverse_transform([pred])[0]
    confidence = prob[pred]
    print(f"Sample {i+1}: {class_name} (confidence: {confidence:.3f})")


# %%
model_package = {
    'model': xgb_model,
    'label_encoder': label_encoder_y,
    'feature_columns': X.columns.tolist(),
    'model_type': 'XGBoost',
    'accuracy': accuracy
}

model_filename = "xgboost_mushrooms_complete.pkl"
joblib.dump(model_package, model_filename)
print(f"✅ Complete model package saved as {model_filename}")
print("Package contains: model, label_encoder, feature_columns, model_type, accuracy")

# %%
print("\n" + "="*50)
print("XGBoost Model Training Summary")
print("="*50)
print(f"Final Accuracy: {accuracy:.4f}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Model package saved: {model_filename}")
print("Contains: model + label_encoder + feature_columns + metadata")
print("="*50)

# %%




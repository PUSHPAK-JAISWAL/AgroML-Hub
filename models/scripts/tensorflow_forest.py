# %%
"""TensorFlow Neural Network for Forest Cover Type Prediction"""

# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential,layers,regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# %%
print("Loading and preprocessing data...")
data = pd.read_csv(r'../datasets/forest.csv')

# %%
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

# %%
data = data.dropna()
print(f"Data shape after preprocessing: {data.shape}")

# %%
data.head()

# %%
# Features and Target
X = data.drop(columns=["Cover_Type"])
y = data["Cover_Type"]


# %%
y = y-1

# %%
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Number of classes: {len(np.unique(y))}")


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
print(f"Training samples: {X_train_scaled.shape[0]}")
print(f"Test samples: {X_test_scaled.shape[0]}")
print(f"Features: {X_train_scaled.shape[1]}")

# %%
model = Sequential()

# %%
# Input layer
model.add(layers.Input(shape=(12,)))

# First hidden layer (largest) - 1024 neurons
model.add(layers.Dense(
    1024, 
    activation='relu',
    kernel_regularizer=regularizers.L1L2()
))
model.add(layers.Dropout(0.3))

# Second hidden layer - 512 neurons
model.add(layers.Dense(
    512, 
    activation='relu',
    kernel_regularizer=regularizers.L1L2()
))
model.add(layers.Dropout(0.4))

# Third hidden layer - 256 neurons
model.add(layers.Dense(
    256, 
    activation='relu',
    kernel_regularizer=regularizers.L1L2()
))
model.add(layers.Dropout(0.4))

# Fourth hidden layer - 128 neurons
model.add(layers.Dense(
    128, 
    activation='relu',
    kernel_regularizer=regularizers.L1L2()
))
model.add(layers.Dropout(0.4))

# Fifth hidden layer - 64 neurons
model.add(layers.Dense(
    64, 
    activation='relu',
    kernel_regularizer=regularizers.L1L2()
))
model.add(layers.Dropout(0.3))

# Sixth hidden layer - 32 neurons
model.add(layers.Dense(
    32, 
    activation='relu',
    kernel_regularizer=regularizers.L1L2()
))
model.add(layers.Dropout(0.2))

# Output layer - 7 classes
model.add(layers.Dense(7, activation='softmax'))


# %%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=50,
    restore_best_weights=True
)

# %%
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %%
print("\nModel Architecture:")
model.summary()

# %%
print("\nTraining the model...")
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=500,
    batch_size=64,
    callbacks=[early_stopping]
)


# %%
print("\nEvaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")


# %%
y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# %%
y_test_original = y_test + 1
y_pred_original = y_pred + 1

# %%
print(f"Accuracy: {accuracy_score(y_test_original, y_pred_original):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_original, y_pred_original))

# %%
sample_indices = range(5)
sample_predictions = y_pred_original[sample_indices]
sample_actual = y_test_original.iloc[sample_indices].tolist()

# %%
print(f"\nSample Predictions: {sample_predictions.tolist()}")
print(f"Actual Values: {sample_actual}")

# %%
model.save("forest_cover_model.keras")
print("✅ TensorFlow model saved as forest_cover_model.keras")

# %%
def representative_dataset():
    for i in range(100):
        idx = np.random.randint(0, len(X_train_scaled))
        sample = X_train_scaled[idx:idx+1].astype(np.float32)
        yield [sample]

# %%
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# %%
tflite_model = converter.convert()

# %%
with open("forest_cover_model_quantized.tflite", "wb") as f:
    f.write(tflite_model)


# %%
print("\nTesting quantized TFLite model accuracy...")
interpreter = tf.lite.Interpreter(model_path="forest_cover_model_quantized.tflite")
interpreter.allocate_tensors()

# %%
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# %%
tflite_predictions = []
for sample in X_test_scaled:
    interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, -1).astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(output_data) + 1  # Convert back to 1-7
    tflite_predictions.append(pred_class)

# %%
tflite_accuracy = accuracy_score(y_test_original, tflite_predictions)
print(f"Original TensorFlow model accuracy: {test_accuracy:.4f}")
print(f"Quantized TFLite model accuracy: {tflite_accuracy:.4f}")
print(f"Accuracy difference: {(test_accuracy - tflite_accuracy):.4f}")

# %%
print(f"\nFirst 10 predictions comparison:")
print(f"Original model: {y_pred_original[:10].tolist()}")
print(f"Quantized model: {tflite_predictions[:10]}")
print(f"Actual values: {y_test_original.iloc[:10].tolist()}")

# %%
import os
keras_size = os.path.getsize("forest_cover_model.keras") / 1024  # KB
tflite_size = os.path.getsize("forest_cover_model_quantized.tflite") / 1024  # KB

# %%
print(f"\nModel Size Comparison:")
print(f"TensorFlow model: {keras_size:.2f} KB")
print(f"Quantized TFLite model: {tflite_size:.2f} KB")
print(f"Size reduction: {((keras_size - tflite_size) / keras_size * 100):.1f}%")



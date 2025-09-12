# %%
import numpy as np
import pandas as pd
import pathlib
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers

# %%
DATA_PATH = "../datasets/mushrooms.csv"   # update path as needed
KERAS_MODEL_PATH = "mushroom_model.keras" # saved Keras model (.keras)
TFLITE_MODEL_PATH = "mushroom_model_quantized.tflite"

# %%
df = pd.read_csv(DATA_PATH)

# %%
cols = df.columns.tolist()
assert 'class' in cols, "CSV must contain 'class' column"

# %%
X_df = df.drop(columns=['class']).copy()
y_ser = df['class'].copy()

# %%
X_enc = X_df.copy()
labelers = {}
for c in X_df.columns:
    le = LabelEncoder()
    X_enc[c] = le.fit_transform(X_df[c].astype(str))
    labelers[c] = le

# %%
le_y = LabelEncoder()
y = le_y.fit_transform(y_ser.astype(str))  # 0/1 array

# %%
X_np = X_enc.values.astype(np.float32)  # raw numeric (label-encoded) inputs
y_np = np.array(y).astype(np.int32)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.20, random_state=42, stratify=y_np
)

# %%
print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %%
def build_inverted_pyramid_with_normalization(input_dim, dropout_rates=None):
    if dropout_rates is None:
        dropout_rates = [0.3, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2]

    norm = layers.Normalization(axis=-1, dtype="float32")
    # We'll adapt norm on the training data below
    inp = layers.Input(shape=(input_dim,), dtype=tf.float32, name="input")
    x = norm(inp)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rates[0])(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rates[1])(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rates[2])(x)

    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rates[3])(x)

    x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rates[4])(x)

    x = layers.Dense(8, activation='relu', kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rates[5])(x)

    x = layers.Dense(4, activation='relu', kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rates[6])(x)

    x = layers.Dense(2, activation='relu', kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rates[7])(x)

    out = layers.Dense(1, activation='sigmoid', dtype="float32")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    # attach normalization layer instance so we can adapt it externally before training
    model.normalization_layer = norm
    return model

# %%
model = build_inverted_pyramid_with_normalization(input_dim=X_train.shape[1])
# adapt normalization on raw (label-encoded) training data
model.normalization_layer.adapt(X_train)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=1000,
    batch_size=64,
    callbacks=[early_stopping]
)

# %%
model.save(KERAS_MODEL_PATH)
print(f"Keras model saved to: {KERAS_MODEL_PATH}")

# %%
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_pred_probs_keras = model.predict(X_test).flatten()
y_pred_keras = (y_pred_probs_keras > 0.5).astype(int)
print("\nKeras model accuracy:", accuracy_score(y_test, y_pred_keras))
print(classification_report(y_test, y_pred_keras, digits=4))

# %%
def representative_dataset():
    for i in range(X_train.shape[0]):
        sample = X_train[i:i+1].astype(np.float32)
        yield [sample]

# %%
# Create converter and quantize with representative dataset (default optimizations)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# NOTE: by not forcing target_spec, converter will choose quantization that fits model & repr. dataset.
tflite_quant = converter.convert()

# %%
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_quant)
print(f"TFLite quantized model saved to: {TFLITE_MODEL_PATH}")

# %%
def evaluate_tflite_model(tflite_file, X_raw, y_true):
    # X_raw: raw integer-encoded inputs (not normalized); model's Normalization layer is inside model
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale, in_zero_point = input_details[0].get('quantization', (0.0, 0))
    out_scale, out_zero_point = output_details[0].get('quantization', (0.0, 0))

    preds = []
    for i in range(X_raw.shape[0]):
        inp = X_raw[i:i+1].astype(np.float32)

        # If input is quantized, we must quantize the float input
        if in_scale and in_scale != 0:
            # quantize
            q_in = (inp / in_scale + in_zero_point).round().astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], q_in)
        else:
            interpreter.set_tensor(input_details[0]['index'], inp.astype(input_details[0]['dtype']))

        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        # dequantize if output quantized
        if out_scale and out_scale != 0:
            out_deq = (out.astype(np.float32) - out_zero_point) * out_scale
        else:
            out_deq = out.astype(np.float32)
        prob = float(out_deq.flatten()[0])
        pred = 1 if prob > 0.5 else 0
        preds.append(pred)

    preds = np.array(preds, dtype=int)
    acc = accuracy_score(y_true, preds)
    print("\nTFLite model evaluation")
    print("Accuracy:", acc)
    print(classification_report(y_true, preds, digits=4))

# %%
evaluate_tflite_model(TFLITE_MODEL_PATH, X_test, y_test)

# %%
print("\nLabel encoder classes for 'class' (target):", list(le_y.classes_))
print("If you want to predict raw labels from model predictions, use le_y.inverse_transform([...])")



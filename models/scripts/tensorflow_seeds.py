# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pathlib

# %%
DATA_PATH = "../datasets/Seed_Data.csv"   # update if necessary
KERAS_MODEL_PATH = "seeds_model.keras"
TFLITE_MODEL_PATH = "seeds_model_quantized.tflite"

# %%
df = pd.read_csv(DATA_PATH)

# %%
print("Loaded:", df.shape)
df = df.dropna()  # safety

# %%
X_df = df.drop(columns=["target"])
y_ser = df["target"]

# %%
X = X_df.values.astype(np.float32)

# %%
le_y = LabelEncoder()
y = le_y.fit_transform(y_ser.astype(str)).astype(np.int32)
num_classes = len(le_y.classes_)
print("Classes:", num_classes, list(le_y.classes_))

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Train/Test shapes:", X_train.shape, X_test.shape)

# %%
def build_inverted_pyramid_multi(input_dim, num_classes):
    inp = layers.Input(shape=(input_dim,), dtype=tf.float32, name="input")
    norm = layers.Normalization(dtype="float32")
    x = norm(inp)   # we'll adapt this layer later

    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(8, activation="relu", kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(4, activation="relu", kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(2, activation="relu", kernel_regularizer=regularizers.L1L2())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # multiclass output: softmax
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    # attach norm for external adapt
    model.normalization_layer = norm
    return model


# %%
model = build_inverted_pyramid_multi(X_train.shape[1], num_classes)
# Adapt normalization on training data
model.normalization_layer.adapt(X_train)

# %%
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
model.summary()

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=500,
    batch_size=32,
    callbacks=[early_stopping]
)

# %%
model.save(KERAS_MODEL_PATH)
print("Saved Keras model to:", KERAS_MODEL_PATH)

# %%
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_probs = model.predict(X_test, verbose=0)  # shape (N, num_classes)
y_pred = np.argmax(y_probs, axis=1)
print("\nKeras model accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

# %%
def representative_dataset_full():
    # yield full training set, one row at a time
    for i in range(X_train.shape[0]):
        sample = X_train[i:i+1].astype(np.float32)   # shape (1, features)
        yield [sample]

# %%
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_full

# %%
print("\nConverting to TFLite (this may take a moment)...")
tflite_model = converter.convert()
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)
print("Saved TFLite quantized model to:", TFLITE_MODEL_PATH)


# %%
def evaluate_tflite_multiclass(tflite_path, X_data, y_true):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale, in_zero_point = input_details[0].get("quantization", (0.0, 0))
    out_scale, out_zero_point = output_details[0].get("quantization", (0.0, 0))

    preds = []
    for i in range(X_data.shape[0]):
        inp = X_data[i:i+1].astype(np.float32)  # shape (1,features)

        # quantize input if required
        if in_scale and in_scale != 0:
            q = (inp / in_scale + in_zero_point).round().astype(input_details[0]["dtype"])
            interpreter.set_tensor(input_details[0]["index"], q)
        else:
            interpreter.set_tensor(input_details[0]["index"], inp.astype(input_details[0]["dtype"]))

        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])  # shape (1, num_classes) possibly quantized

        # dequantize output if needed
        if out_scale and out_scale != 0:
            out_deq = (out.astype(np.float32) - out_zero_point) * out_scale
        else:
            out_deq = out.astype(np.float32)

        pred = int(np.argmax(out_deq, axis=1)[0])
        preds.append(pred)

    preds = np.array(preds, dtype=int)
    print("\nTFLite evaluation:")
    print("Accuracy:", accuracy_score(y_true, preds))
    print(classification_report(y_true, preds, digits=4))

# %%
evaluate_tflite_multiclass(TFLITE_MODEL_PATH, X_test, y_test)

# 11) Done: print helpful info
print("\nDone. Keras saved to:", KERAS_MODEL_PATH, "TFLite saved to:", TFLITE_MODEL_PATH)
print("Use le_y.inverse_transform([pred]) to map numeric labels back to original class values.")

# %%




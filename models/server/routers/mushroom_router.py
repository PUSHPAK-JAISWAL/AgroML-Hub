# routers/mushroom_router.py
"""
Mushroom prediction router for FastAPI.

Endpoints:
- POST  /mushroom/predict        -> single sample prediction
- POST  /mushroom/predict_batch  -> batch prediction
- GET   /mushroom/mappings       -> view categorical mappings (human-friendly)
- POST  /mushroom/mappings      -> update/extend mappings at runtime (in-memory)
- GET   /mushroom/info          -> model & feature info (paths + loaded status)

This router will attempt to load the original CSV to reconstruct LabelEncoders
so textual categorical inputs (single-letter codes or full descriptions) are
mapped to the same numeric encodings used during training.
"""
from typing import List, Union, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

router = APIRouter(prefix="/mushroom", tags=["mushroom"])

# ---------------------------
# Feature ordering (must match training)
# ---------------------------
FEATURE_NAMES = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat"
]
EXPECTED_FEATURE_DIM = len(FEATURE_NAMES)  # 22

# ---------------------------
# The descriptive mappings you used during preprocessing (letter -> description)
# This mirrors the 'mappings' dict from your training script.
# Keys are lowercased / normalized descriptions (we'll use them to accept descriptive words).
# ---------------------------
MAPPINGS_LETTER_TO_DESC: Dict[str, Dict[str, str]] = {
    "cap-shape": {
        "b": "bell", "c": "conical", "x": "convex",
        "f": "flat", "k": "knobbed", "s": "sunken"
    },
    "cap-surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
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

# lower-case descriptive mapping convenience (letter->desc already lower-case)
# We'll also provide reverse mapping from description -> letter
DESC_TO_LETTER: Dict[str, Dict[str, str]] = {}
for col, d in MAPPINGS_LETTER_TO_DESC.items():
    DESC_TO_LETTER[col] = {v.lower(): k for k, v in d.items()}

# ---------------------------
# Pydantic request models
# ---------------------------
class SingleSample(BaseModel):
    sample: Union[Dict[str, Union[str, float, int]], List[Union[str, float, int]]]

class BatchSamples(BaseModel):
    samples: List[Union[Dict[str, Union[str, float, int]], List[Union[str, float, int]]]]

class MappingsUpdate(BaseModel):
    add_mappings: Optional[Dict[str, Dict[str, str]]] = None

# ---------------------------
# Model paths (adjust if required)
# ---------------------------
RF_PATH = 'E:/AgroML_Hub/models/scikit_models/random_mushrooms_model.pkl'
XGB_PATH =  'E:/AgroML_Hub/models/xgboost_models/xgboost_mushrooms_complete.pkl'
TF_PATH =  'E:/AgroML_Hub/models/tensorflow_models/mushroom_model.keras'
TFLITE_PATH =  'E:/AgroML_Hub/models/tensorflow_quantized_model/mushroom_model_quantized.tflite'

# ---------------------------
# Load helpers + logs
# ---------------------------
LOAD_LOGS: Dict[str, str] = {}

def _load_joblib_safe(path: str):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        msg = f"file not found: {abs_path}"
        print(f"[mushroom.models] {msg}")
        LOAD_LOGS[abs_path] = msg
        return None
    try:
        obj = joblib.load(abs_path)
        msg = f"loaded (type={type(obj)})"
        print(f"[mushroom.models] {abs_path} -> {msg}")
        LOAD_LOGS[abs_path] = msg
        return obj
    except Exception as e:
        msg = f"failed to load: {repr(e)}"
        print(f"[mushroom.models] {abs_path} -> {msg}")
        LOAD_LOGS[abs_path] = msg
        return None

# ---------------------------
# Attempt to construct LabelEncoders by loading the original dataset
# The router will try a few common dataset locations.
# ---------------------------
_DATASET_PATHS = [
    os.path.abspath(os.path.join("datasets", "mushrooms.csv")),
    os.path.abspath(os.path.join("..", "datasets", "mushrooms.csv")),
    os.path.abspath(os.path.join(".", "datasets", "mushrooms.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "datasets", "mushrooms.csv"))
]

_label_encoders: Dict[str, LabelEncoder] = {}
_label_encoder_target: Optional[LabelEncoder] = None
_dataset_loaded = False

def _try_build_encoders():
    global _label_encoders, _label_encoder_target, _dataset_loaded
    for p in _DATASET_PATHS:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                # If the dataset uses single-letter codes as in original dataset,
                # we should replace the single-letter codes with human descriptions
                # exactly like the training script did. Use MAPPINGS_LETTER_TO_DESC.
                df_local = df.copy()
                # apply replacement only for columns we have mappings for
                replace_map = {}
                for col in FEATURE_NAMES + ["class"]:
                    if col in df_local.columns and col in MAPPINGS_LETTER_TO_DESC:
                        replace_map[col] = MAPPINGS_LETTER_TO_DESC[col]
                if replace_map:
                    df_local = df_local.replace(replace_map)

                # build LabelEncoders for features
                for col in FEATURE_NAMES:
                    if col in df_local.columns:
                        le = LabelEncoder()
                        le.fit(df_local[col].astype(str))
                        _label_encoders[col] = le

                # target encoder
                if "class" in df_local.columns:
                    le_y = LabelEncoder()
                    le_y.fit(df_local["class"].astype(str))
                    _label_encoder_target = le_y
                    # store in module-level var
                    globals()["_label_encoder_target"] = le_y

                _dataset_loaded = True
                print(f"[mushroom.encoders] built encoders from {p}")
                LOAD_LOGS[os.path.abspath(p)] = "encoders built from dataset"
                return
            except Exception as e:
                print(f"[mushroom.encoders] failed to build encoders from {p}: {e}")
                LOAD_LOGS[os.path.abspath(p)] = f"failed to build encoders: {repr(e)}"
    # if we reach here, no dataset loaded
    print("[mushroom.encoders] no dataset found to build encoders; API will still accept numeric encodings only.")
    _dataset_loaded = False

_try_build_encoders()

# ---------------------------
# Load models
# ---------------------------
_rf_model = _load_joblib_safe(RF_PATH)

_xgb_raw = _load_joblib_safe(XGB_PATH)
_xgb_model = None
_xgb_feature_columns = None
_xgb_label_encoder = None
if _xgb_raw is not None:
    if isinstance(_xgb_raw, dict):
        _xgb_model = _xgb_raw.get("model")
        _xgb_feature_columns = _xgb_raw.get("feature_columns")
        _xgb_label_encoder = _xgb_raw.get("label_encoder")  # may be present
    else:
        _xgb_model = _xgb_raw

_tf_model = None
if os.path.exists(TF_PATH):
    try:
        _tf_model = tf.keras.models.load_model(TF_PATH)
        LOAD_LOGS[os.path.abspath(TF_PATH)] = "loaded (tf keras model)"
        print(f"[mushroom.models] loaded TF model: {os.path.abspath(TF_PATH)}")
    except Exception as e:
        LOAD_LOGS[os.path.abspath(TF_PATH)] = f"failed to load tf model: {repr(e)}"
        print(f"[mushroom.models] {os.path.abspath(TF_PATH)} -> failed to load: {repr(e)}")
        _tf_model = None
else:
    LOAD_LOGS[os.path.abspath(TF_PATH)] = f"file not found: {os.path.abspath(TF_PATH)}"

_tflite_interpreter = None
_tflite_input_details = None
_tflite_output_details = None
if os.path.exists(TFLITE_PATH):
    try:
        _tflite_interpreter = tf.lite.Interpreter(model_path=os.path.abspath(TFLITE_PATH))
        _tflite_interpreter.allocate_tensors()
        _tflite_input_details = _tflite_interpreter.get_input_details()
        _tflite_output_details = _tflite_interpreter.get_output_details()
        LOAD_LOGS[os.path.abspath(TFLITE_PATH)] = "loaded (tflite interpreter)"
        print(f"[mushroom.models] loaded TFLite: {os.path.abspath(TFLITE_PATH)}")
    except Exception as e:
        LOAD_LOGS[os.path.abspath(TFLITE_PATH)] = f"failed to load tflite: {repr(e)}"
        print(f"[mushroom.models] {os.path.abspath(TFLITE_PATH)} -> failed to load: {repr(e)}")
        _tflite_interpreter = None
else:
    LOAD_LOGS[os.path.abspath(TFLITE_PATH)] = f"file not found: {os.path.abspath(TFLITE_PATH)}"

# ---------------------------
# Helpers to convert textual input to numeric encoding
# ---------------------------
def _encode_feature_value(col: str, value: Union[str, int, float]) -> int:
    """
    Convert a feature value (string/int/float) to the numeric encoding used
    by the training LabelEncoders. Accepts:
      - numeric (int/float) -> returns int(value)
      - numeric string -> parsed to int
      - single-letter code (e.g. 'x', 'b') -> translates to descriptive word via MAPPINGS_LETTER_TO_DESC then encodes
      - descriptive text (e.g. 'convex' or 'Convex') -> encoded by the LabelEncoder fitted from dataset
    If label-encoders are not available, numeric inputs are accepted only.
    """
    # numeric case
    import numpy as _np
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        # numeric-string
        if s.isdigit():
            return int(s)
        # try direct encoding using encoder if available
        le = _label_encoders.get(col)
        if le is not None:
            # Try exact string first (case-sensitive), then lower-case variant, then mapped letter->desc
            try:
                return int(le.transform([s])[0])
            except Exception:
                try:
                    return int(le.transform([s.lower()])[0])
                except Exception:
                    # if single-letter code provided, map to desc then encode
                    if len(s) == 1 and col in MAPPINGS_LETTER_TO_DESC:
                        letter = s.lower()
                        desc = MAPPINGS_LETTER_TO_DESC.get(col, {}).get(letter)
                        if desc is not None:
                            try:
                                return int(le.transform([desc])[0])
                            except Exception:
                                pass
                    # maybe given description, try lower-case mapping via desc->letter -> desc again
                    if col in DESC_TO_LETTER:
                        # try mapping synonyms (like uppercase first letter)
                        try:
                            return int(le.transform([s.capitalize()])[0])
                        except Exception:
                            pass
        # If no encoder or all attempts failed, try letter->desc fallback (return the index of desc in sorted unique)
        if len(s) == 1 and col in MAPPINGS_LETTER_TO_DESC:
            # try to return a stable mapping based on ordering of mapping keys
            letter = s.lower()
            desc = MAPPINGS_LETTER_TO_DESC[col].get(letter)
            if desc is not None:
                # if encoder exists, we already tried; otherwise can't produce numeric reliably
                le = _label_encoders.get(col)
                if le is not None:
                    try:
                        return int(le.transform([desc])[0])
                    except Exception:
                        pass
        # final fallback: cannot encode textual value
        raise HTTPException(status_code=422, detail=f"Unknown or unencodable value for {col}: {value}")
    raise HTTPException(status_code=422, detail=f"Unsupported type for feature {col}: {type(value)}")

# ---------------------------
# Convert sample to ordered numpy array
# ---------------------------
def _to_ordered_array(sample: Union[Dict[str, Union[str, float, int]], List[Union[str, float, int]]]) -> np.ndarray:
    if isinstance(sample, dict):
        arr = []
        for name in FEATURE_NAMES:
            if name not in sample:
                raise HTTPException(status_code=422, detail=f"Missing feature: {name}")
            val = sample[name]
            # encode categorical via encoder if needed (dataset features are categorical)
            try:
                encoded = _encode_feature_value(name, val)
                arr.append(float(encoded))
            except HTTPException as e:
                # If dataset encoders not available, still allow numeric values
                if isinstance(val, (int, float, np.integer)):
                    arr.append(float(val))
                else:
                    raise e
        return np.array(arr, dtype=np.float32)

    elif isinstance(sample, list):
        if len(sample) != EXPECTED_FEATURE_DIM:
            raise HTTPException(status_code=422, detail=f"Expected list length {EXPECTED_FEATURE_DIM}, got {len(sample)}")
        arr = []
        for idx, val in enumerate(sample):
            col = FEATURE_NAMES[idx]
            try:
                encoded = _encode_feature_value(col, val)
                arr.append(float(encoded))
            except HTTPException as e:
                if isinstance(val, (int, float, np.integer)):
                    arr.append(float(val))
                else:
                    raise e
        return np.array(arr, dtype=np.float32)
    else:
        raise HTTPException(status_code=422, detail="Sample must be a dict (feature->value) or a list of values.")

# ---------------------------
# scaling helper (no external scaler)
# ---------------------------
def _apply_scaler(X: np.ndarray) -> np.ndarray:
    return X

# ---------------------------
# Prediction helper wrappers
# ---------------------------
def _predict_rf(x: np.ndarray) -> Dict[str, Any]:
    if _rf_model is None:
        raise HTTPException(status_code=500, detail="RandomForest model not loaded.")
    # Prefer DataFrame with feature names to avoid warnings and match training shape
    try:
        df = pd.DataFrame(x, columns=FEATURE_NAMES)
        preds = _rf_model.predict(df)
        probs = _rf_model.predict_proba(df).tolist() if hasattr(_rf_model, "predict_proba") else None
    except Exception:
        preds = _rf_model.predict(x)
        probs = _rf_model.predict_proba(x).tolist() if hasattr(_rf_model, "predict_proba") else None
    return {"predictions": preds.tolist(), "probabilities": probs}

def _predict_xgb(x: np.ndarray) -> Dict[str, Any]:
    if _xgb_model is None:
        raise HTTPException(status_code=500, detail="XGBoost model not loaded.")
    try:
        cols = _xgb_feature_columns if _xgb_feature_columns is not None else FEATURE_NAMES
        df = pd.DataFrame(x, columns=cols)
        preds = _xgb_model.predict(df)
        probs = _xgb_model.predict_proba(df).tolist() if hasattr(_xgb_model, "predict_proba") else None
    except Exception:
        preds = _xgb_model.predict(x)
        probs = _xgb_model.predict_proba(x).tolist() if hasattr(_xgb_model, "predict_proba") else None
    return {"predictions": preds.tolist(), "probabilities": probs}

def _predict_tf(x: np.ndarray) -> Dict[str, Any]:
    if _tf_model is None:
        raise HTTPException(status_code=500, detail="TensorFlow model not loaded.")
    yp = _tf_model.predict(x, verbose=0)
    # model used sigmoid -> shape (N,1)
    if yp.ndim == 2 and yp.shape[1] == 1:
        probs = yp.flatten().tolist()
        preds = [1 if p > 0.5 else 0 for p in probs]
        # return 0/1 preds (matching LabelEncoder encoding)
        return {"predictions": preds, "probabilities": [[1 - p, p] for p in probs]}
    else:
        # fallback: if model outputs two-class softmax
        if yp.ndim == 2 and yp.shape[1] == 2:
            preds = list(np.argmax(yp, axis=1))
            return {"predictions": preds, "probabilities": yp.tolist()}
        # else unknown shape
        raise HTTPException(status_code=500, detail="Unexpected TF model output shape")

def _predict_tflite(x: np.ndarray) -> Dict[str, Any]:
    if _tflite_interpreter is None:
        raise HTTPException(status_code=500, detail="TFLite model not loaded.")
    results = []
    probs = []
    for sample in x:
        s = sample.reshape(1, -1).astype(_tflite_input_details[0]["dtype"])
        _tflite_interpreter.set_tensor(_tflite_input_details[0]["index"], s)
        _tflite_interpreter.invoke()
        output_data = _tflite_interpreter.get_tensor(_tflite_output_details[0]["index"])
        # If output is scalar prob
        if output_data.ndim == 2 and output_data.shape[1] == 1:
            p = float(output_data.flatten()[0])
            pred = 1 if p > 0.5 else 0
            results.append(int(pred))
            probs.append([1 - p, p])
        elif output_data.ndim == 2 and output_data.shape[1] == 2:
            pred = int(np.argmax(output_data, axis=1)[0])
            results.append(pred)
            probs.append(output_data.tolist()[0])
        else:
            # fallback: take argmax flatten
            pred = int(np.argmax(output_data))
            results.append(pred)
            probs.append(output_data.tolist()[0] if hasattr(output_data, "tolist") else [float(output_data)])
    return {"predictions": results, "probabilities": probs}

# ---------------------------
# Target label utility
# ---------------------------
def _label_for_target(pred_int: int) -> str:
    """
    Convert numeric prediction 0/1 to human readable label using:
     - xgb packaged label_encoder if present
     - global target encoder built from dataset if available
     - fallback: common mapping (0 -> edible, 1 -> poisonous) — try to reflect training order
    """
    # check xgb packaged label encoder first
    if _xgb_label_encoder is not None:
        try:
            return str(_xgb_label_encoder.inverse_transform([pred_int])[0])
        except Exception:
            pass
    # global target encoder
    le = globals().get("_label_encoder_target")
    if le is not None:
        try:
            val = le.inverse_transform([pred_int])[0]
            return str(val)
        except Exception:
            pass
    # fallback common mapping: many pipelines encode 'edible'->0, 'poisonous'->1
    fallback_map = {0: "edible", 1: "poisonous"}
    return fallback_map.get(int(pred_int), str(pred_int))

# ---------------------------
# Endpoints
# ---------------------------
@router.post("/predict", summary="Predict edible/poisonous for a single mushroom sample")
def predict(sample: SingleSample):
    arr = _to_ordered_array(sample.sample).reshape(1, -1)
    arr_scaled = _apply_scaler(arr)

    out: Dict[str, Any] = {}

    # RF
    try:
        rf_out = _predict_rf(arr)
        rf_pred = int(rf_out["predictions"][0])
        out["randomforest"] = {
            "prediction": rf_pred,
            "label": _label_for_target(rf_pred),
            "probabilities": rf_out["probabilities"][0] if rf_out["probabilities"] else None
        }
    except HTTPException as e:
        out["randomforest"] = {"error": str(e.detail)}

    # XGB
    try:
        xgb_out = _predict_xgb(arr)
        xgb_pred = int(xgb_out["predictions"][0])
        out["xgboost"] = {
            "prediction": xgb_pred,
            "label": _label_for_target(xgb_pred),
            "probabilities": xgb_out["probabilities"][0] if xgb_out["probabilities"] else None
        }
    except HTTPException as e:
        out["xgboost"] = {"error": str(e.detail)}

    # TensorFlow
    try:
        tf_out = _predict_tf(arr_scaled)
        tf_pred = int(tf_out["predictions"][0])
        out["tensorflow"] = {
            "prediction": tf_pred,
            "label": _label_for_target(tf_pred),
            "probabilities": tf_out["probabilities"][0]
        }
    except HTTPException as e:
        out["tensorflow"] = {"error": str(e.detail)}

    # TFLite
    try:
        tflite_out = _predict_tflite(arr_scaled)
        tflite_pred = int(tflite_out["predictions"][0])
        out["quantized"] = {
            "prediction": tflite_pred,
            "label": _label_for_target(tflite_pred),
            "probabilities": tflite_out["probabilities"][0]
        }
    except HTTPException as e:
        out["quantized"] = {"error": str(e.detail)}

    return out

@router.post("/predict_batch", summary="Predict edible/poisonous for a batch of mushroom samples")
def predict_batch(batch: BatchSamples):
    prepared = []
    for s in batch.samples:
        prepared.append(_to_ordered_array(s))
    X = np.stack(prepared, axis=0)
    X_scaled = _apply_scaler(X)

    out: Dict[str, Any] = {}

    # RF
    try:
        rf_out = _predict_rf(X)
        rf_preds = [int(x) for x in rf_out["predictions"]]
        out["randomforest"] = {
            "predictions": rf_preds,
            "labels": [_label_for_target(x) for x in rf_preds],
            "probabilities": rf_out["probabilities"]
        }
    except HTTPException as e:
        out["randomforest"] = {"error": str(e.detail)}

    # XGB
    try:
        xgb_out = _predict_xgb(X)
        xgb_preds = [int(x) for x in xgb_out["predictions"]]
        out["xgboost"] = {
            "predictions": xgb_preds,
            "labels": [_label_for_target(x) for x in xgb_preds],
            "probabilities": xgb_out["probabilities"]
        }
    except HTTPException as e:
        out["xgboost"] = {"error": str(e.detail)}

    # TensorFlow
    try:
        tf_out = _predict_tf(X_scaled)
        tf_preds = [int(x) for x in tf_out["predictions"]]
        out["tensorflow"] = {
            "predictions": tf_preds,
            "labels": [_label_for_target(x) for x in tf_preds],
            "probabilities": tf_out["probabilities"]
        }
    except HTTPException as e:
        out["tensorflow"] = {"error": str(e.detail)}

    # TFLite
    try:
        tflite_out = _predict_tflite(X_scaled)
        tflite_preds = [int(x) for x in tflite_out["predictions"]]
        out["quantized"] = {
            "predictions": tflite_preds,
            "labels": [_label_for_target(x) for x in tflite_preds],
            "probabilities": tflite_out["probabilities"]
        }
    except HTTPException as e:
        out["quantized"] = {"error": str(e.detail)}

    return out

# ---------------------------
# Mappings endpoints
# ---------------------------
@router.get("/mappings", summary="Get current descriptive mappings (letter -> description)")
def get_mappings():
    return {
        "letter_to_description": MAPPINGS_LETTER_TO_DESC,
        "description_to_letter": DESC_TO_LETTER,
        "dataset_encoders_available": _dataset_loaded,
        "encoded_label_counts": {k: len(v.classes_) for k, v in _label_encoders.items()} if _dataset_loaded else {}
    }

@router.post("/mappings", summary="Add/extend descriptive mappings (in-memory only)")
def update_mappings(payload: MappingsUpdate = Body(...)):
    """
    Payload example:
    {
      "add_mappings": {
         "cap-color": {"z":"azure"},
         "new_col": {"a":"something"}
      }
    }
    This will extend the in-memory MAPPINGS_LETTER_TO_DESC and DESC_TO_LETTER.
    """
    updated = {"added": []}
    if payload.add_mappings:
        for col, mapping in payload.add_mappings.items():
            col = col.strip()
            if col not in MAPPINGS_LETTER_TO_DESC:
                MAPPINGS_LETTER_TO_DESC[col] = {}
                DESC_TO_LETTER[col] = {}
            for letter, desc in mapping.items():
                MAPPINGS_LETTER_TO_DESC[col][letter] = desc
                DESC_TO_LETTER[col][desc.lower()] = letter
                updated["added"].append({col: {letter: desc}})
    return {"status": "ok", "updated": updated}

# ---------------------------
# Info endpoint
# ---------------------------
@router.get("/info", summary="Model & feature info")
def info():
    return {
        "feature_names": FEATURE_NAMES,
        "expected_feature_dim": EXPECTED_FEATURE_DIM,
        "dataset_encoders_built": _dataset_loaded,
        "models": {
            "randomforest": {
                "loaded": _rf_model is not None,
                "path": os.path.abspath(RF_PATH),
                "log": LOAD_LOGS.get(os.path.abspath(RF_PATH), "")
            },
            "xgboost": {
                "loaded": _xgb_model is not None,
                "path": os.path.abspath(XGB_PATH),
                "log": LOAD_LOGS.get(os.path.abspath(XGB_PATH), "")
            },
            "tensorflow": {
                "loaded": _tf_model is not None,
                "path": os.path.abspath(TF_PATH),
                "log": LOAD_LOGS.get(os.path.abspath(TF_PATH), "")
            },
            "quantized_tflite": {
                "loaded": _tflite_interpreter is not None,
                "path": os.path.abspath(TFLITE_PATH),
                "log": LOAD_LOGS.get(os.path.abspath(TFLITE_PATH), "")
            }
        }
    }

# routers/seeds_router.py
"""
Seeds (wheat kernels) router for FastAPI.

Endpoints:
- POST  /seeds/predict        -> single sample prediction
- POST  /seeds/predict_batch  -> batch prediction
- GET   /seeds/mappings       -> (kept for parity with forest router)
- POST  /seeds/mappings       -> update mappings (in-memory)
- GET   /seeds/info           -> model & feature info (paths + loaded status)

Expected model files (adjust paths below if different):
- models/scikit_models/random_Seeds_model.pkl
- models/xgboost_models/xgboost_wheat_seeds_complete.pkl  (or plain xgb model)
- models/tensorflow_models/seeds_model.keras
- models/tensorflow_quantized_model/seeds_model_quantized.tflite
"""
from typing import List, Union, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
import joblib
import os
import numpy as np
import tensorflow as tf
import pandas as pd

router = APIRouter(prefix="/seeds", tags=["seeds"])

# ---------------------------
# Feature ordering (must match training)
# ---------------------------
FEATURE_NAMES = [
    "Area",
    "Perimeter",
    "Compactness",
    "length_of_kernel",
    "width_of_kernel",
    "asymetric_coef",
    "length_of_kernel_groove"
]
EXPECTED_FEATURE_DIM = len(FEATURE_NAMES)  # 7

# ---------------------------
# Human-readable labels (three varieties)
# ---------------------------
LABEL_MAP: Dict[int, str] = {
    1: "Kama",
    2: "Rosa",
    3: "Canadian"
}

def _label_from_int(i: int) -> str:
    """
    Accept both 1..3 and 0..2 encodings:
      - if model returns 1..3, map directly
      - if model returns 0..2, map using +1
    Otherwise return the integer as string.
    """
    i = int(i)
    if i in LABEL_MAP:
        return LABEL_MAP[i]
    if (i + 1) in LABEL_MAP:
        return LABEL_MAP[i + 1]
    return str(i)

# ---------------------------
# Mappings (kept for parity with forest router)
# ---------------------------
WILDERNESS_MAP: Dict[str, int] = {}
SOIL_MAP: Dict[str, int] = {}

# ---------------------------
# Pydantic request models
# ---------------------------
class SingleSample(BaseModel):
    sample: Union[Dict[str, Union[str, float, int]], List[Union[str, float, int]]]

class BatchSamples(BaseModel):
    samples: List[Union[Dict[str, Union[str, float, int]], List[Union[str, float, int]]]]

class MappingsUpdate(BaseModel):
    wilderness_map: Optional[Dict[str, int]] = None
    soil_map: Optional[Dict[str, int]] = None

# ---------------------------
# Model file paths (adjust if needed)
# ---------------------------
# By default look for models under ./models/...
RF_PATH = 'E:/AgroML_Hub/models/scikit_models/random_Seeds_model.pkl'
XGB_PATH =  'E:/AgroML_Hub/models/xgboost_models/xgboost_wheat_seeds_complete.pkl'
TF_PATH =  'E:/AgroML_Hub/models/tensorflow_models/seeds_model.keras'
TFLITE_PATH =  'E:/AgroML_Hub/models/tensorflow_quantized_model/seeds_model_quantized.tflite'

# No scaler by default
SCALER_PATH = None

# ---------------------------
# Loader + logs
# ---------------------------
LOAD_LOGS: Dict[str, str] = {}

def _load_joblib_safe(path: str):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        msg = f"file not found: {abs_path}"
        print(f"[seeds.models] {msg}")
        LOAD_LOGS[abs_path] = msg
        return None
    try:
        obj = joblib.load(abs_path)
        msg = f"loaded (type={type(obj)})"
        print(f"[seeds.models] {abs_path} -> {msg}")
        LOAD_LOGS[abs_path] = msg
        return obj
    except Exception as e:
        msg = f"failed to load: {repr(e)}"
        print(f"[seeds.models] {abs_path} -> {msg}")
        LOAD_LOGS[abs_path] = msg
        return None

# ---------------------------
# Load models
# ---------------------------
_rf_model = _load_joblib_safe(RF_PATH)

_xgb_raw = _load_joblib_safe(XGB_PATH)  # may be a dict package or plain model
_xgb_model = None
_xgb_feature_columns = None
if _xgb_raw is not None:
    if isinstance(_xgb_raw, dict) and "model" in _xgb_raw:
        _xgb_model = _xgb_raw["model"]
        _xgb_feature_columns = _xgb_raw.get("feature_columns")
    else:
        _xgb_model = _xgb_raw

_scaler = None  # no scaler currently

_tf_model = None
if os.path.exists(TF_PATH):
    try:
        _tf_model = tf.keras.models.load_model(TF_PATH)
        LOAD_LOGS[os.path.abspath(TF_PATH)] = "loaded (tf keras model)"
        print(f"[seeds.models] loaded TF model: {os.path.abspath(TF_PATH)}")
    except Exception as e:
        LOAD_LOGS[os.path.abspath(TF_PATH)] = f"failed to load tf model: {repr(e)}"
        print(f"[seeds.models] {os.path.abspath(TF_PATH)} -> failed to load: {repr(e)}")
        _tf_model = None
else:
    LOAD_LOGS[os.path.abspath(TF_PATH)] = f"file not found: {os.path.abspath(TF_PATH)}"
    print(f"[seeds.models] file not found: {os.path.abspath(TF_PATH)}")

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
        print(f"[seeds.models] loaded TFLite: {os.path.abspath(TFLITE_PATH)}")
    except Exception as e:
        LOAD_LOGS[os.path.abspath(TFLITE_PATH)] = f"failed to load tflite: {repr(e)}"
        print(f"[seeds.models] {os.path.abspath(TFLITE_PATH)} -> failed to load: {repr(e)}")
        _tflite_interpreter = None
else:
    LOAD_LOGS[os.path.abspath(TFLITE_PATH)] = f"file not found: {os.path.abspath(TFLITE_PATH)}"
    print(f"[seeds.models] file not found: {os.path.abspath(TFLITE_PATH)}")

# ---------------------------
# Input conversion helpers
# ---------------------------
def _to_ordered_array(sample: Union[Dict[str, Union[str, float, int]], List[Union[str, float, int]]]) -> np.ndarray:
    """
    Convert input (dict or list) -> numpy array shape (EXPECTED_FEATURE_DIM,)
    - For dict: read FEATURE_NAMES order
    - For list: must be EXACT length (7)
    """
    if isinstance(sample, dict):
        arr = []
        for name in FEATURE_NAMES:
            if name not in sample:
                raise HTTPException(status_code=422, detail=f"Missing feature: {name}")
            val = sample[name]
            try:
                arr.append(float(val))
            except Exception:
                raise HTTPException(status_code=422, detail=f"Feature {name} must be numeric, got {val}")
        return np.array(arr, dtype=np.float32)

    elif isinstance(sample, list):
        if len(sample) != EXPECTED_FEATURE_DIM:
            raise HTTPException(status_code=422, detail=f"Expected list length {EXPECTED_FEATURE_DIM}, got {len(sample)}")
        arr = []
        for idx, val in enumerate(sample):
            try:
                arr.append(float(val))
            except Exception:
                raise HTTPException(status_code=422, detail=f"Feature {FEATURE_NAMES[idx]} must be numeric, got {val}")
        return np.array(arr, dtype=np.float32)
    else:
        raise HTTPException(status_code=422, detail="Sample must be a dict (feature->value) or a list of numbers/strings.")

def _apply_scaler(X: np.ndarray) -> np.ndarray:
    # no scaler for now
    return X

# ---------------------------
# Prediction wrappers
# ---------------------------
def _predict_rf(x: np.ndarray) -> Dict[str, Any]:
    if _rf_model is None:
        raise HTTPException(status_code=500, detail="RandomForest model not loaded.")
    try:
        df = pd.DataFrame(x, columns=FEATURE_NAMES)
        pred = _rf_model.predict(df)
        probs = _rf_model.predict_proba(df).tolist() if hasattr(_rf_model, "predict_proba") else None
    except Exception:
        pred = _rf_model.predict(x)
        probs = _rf_model.predict_proba(x).tolist() if hasattr(_rf_model, "predict_proba") else None
    return {"predictions": pred.tolist(), "probabilities": probs}

def _predict_xgb(x: np.ndarray) -> Dict[str, Any]:
    if _xgb_model is None:
        raise HTTPException(status_code=500, detail="XGBoost model not loaded.")
    try:
        cols = _xgb_feature_columns if _xgb_feature_columns is not None else FEATURE_NAMES
        df = pd.DataFrame(x, columns=cols)
        pred = _xgb_model.predict(df)
        probs = _xgb_model.predict_proba(df).tolist() if hasattr(_xgb_model, "predict_proba") else None
    except Exception:
        pred = _xgb_model.predict(x)
        probs = _xgb_model.predict_proba(x).tolist() if hasattr(_xgb_model, "predict_proba") else None
    return {"predictions": pred.tolist(), "probabilities": probs}

def _predict_tf(x: np.ndarray) -> Dict[str, Any]:
    if _tf_model is None:
        raise HTTPException(status_code=500, detail="TensorFlow model not loaded.")
    yp = _tf_model.predict(x, verbose=0)
    classes = np.argmax(yp, axis=1)  # likely 0..2 in your training
    return {"predictions": classes.tolist(), "probabilities": yp.tolist()}

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
        pred_class = int(np.argmax(output_data, axis=1)[0])
        results.append(pred_class)
        probs.append(output_data.tolist()[0])
    return {"predictions": results, "probabilities": probs}

# ---------------------------
# Endpoints
# ---------------------------
@router.post("/predict", summary="Predict seed variety for a single sample")
def predict(sample: SingleSample):
    arr = _to_ordered_array(sample.sample).reshape(1, -1)
    arr_scaled = _apply_scaler(arr)

    out: Dict[str, Any] = {}

    # RandomForest
    try:
        rf_out = _predict_rf(arr)
        rf_pred = int(rf_out["predictions"][0])
        out["randomforest"] = {
            "prediction": rf_pred,
            "label": _label_from_int(rf_pred),
            "probabilities": rf_out["probabilities"][0] if rf_out["probabilities"] else None
        }
    except HTTPException as e:
        out["randomforest"] = {"error": str(e.detail)}

    # XGBoost
    try:
        xgb_out = _predict_xgb(arr)
        xgb_pred = int(xgb_out["predictions"][0])
        out["xgboost"] = {
            "prediction": xgb_pred,
            "label": _label_from_int(xgb_pred),
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
            "label": _label_from_int(tf_pred),
            "probabilities": tf_out["probabilities"][0]
        }
    except HTTPException as e:
        out["tensorflow"] = {"error": str(e.detail)}

    # Quantized TFLite
    try:
        tflite_out = _predict_tflite(arr_scaled)
        tflite_pred = int(tflite_out["predictions"][0])
        out["quantized"] = {
            "prediction": tflite_pred,
            "label": _label_from_int(tflite_pred),
            "probabilities": tflite_out["probabilities"][0]
        }
    except HTTPException as e:
        out["quantized"] = {"error": str(e.detail)}

    return out

@router.post("/predict_batch", summary="Predict seed variety for a batch of samples")
def predict_batch(batch: BatchSamples):
    prepared = [_to_ordered_array(s) for s in batch.samples]
    X = np.stack(prepared, axis=0)
    X_scaled = _apply_scaler(X)

    out: Dict[str, Any] = {}

    # RandomForest
    try:
        rf_out = _predict_rf(X)
        rf_preds = [int(x) for x in rf_out["predictions"]]
        out["randomforest"] = {
            "predictions": rf_preds,
            "labels": [_label_from_int(x) for x in rf_preds],
            "probabilities": rf_out["probabilities"]
        }
    except HTTPException as e:
        out["randomforest"] = {"error": str(e.detail)}

    # XGBoost
    try:
        xgb_out = _predict_xgb(X)
        xgb_preds = [int(x) for x in xgb_out["predictions"]]
        out["xgboost"] = {
            "predictions": xgb_preds,
            "labels": [_label_from_int(x) for x in xgb_preds],
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
            "labels": [_label_from_int(x) for x in tf_preds],
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
            "labels": [_label_from_int(x) for x in tflite_preds],
            "probabilities": tflite_out["probabilities"]
        }
    except HTTPException as e:
        out["quantized"] = {"error": str(e.detail)}

    return out

# ---------------------------
# Mapping management endpoints (parity with forest router)
# ---------------------------
@router.get("/mappings", summary="Get current categorical mappings (placeholder for seeds)")
def get_mappings():
    return {"wilderness_map": WILDERNESS_MAP, "soil_map": SOIL_MAP}

@router.post("/mappings", summary="Update/extend categorical mappings (in-memory)")
def update_mappings(payload: MappingsUpdate = Body(...)):
    updated = {"wilderness_added": [], "soil_added": []}
    if payload.wilderness_map:
        for k, v in payload.wilderness_map.items():
            key = k.strip().lower()
            WILDERNESS_MAP[key] = int(v)
            updated["wilderness_added"].append({key: int(v)})
    if payload.soil_map:
        for k, v in payload.soil_map.items():
            key = k.strip().lower()
            SOIL_MAP[key] = int(v)
            updated["soil_added"].append({key: int(v)})
    return {"status": "ok", "updated": updated}

# ---------------------------
# Info endpoint
# ---------------------------
@router.get("/info", summary="Model & feature info")
def info():
    return {
        "feature_names": FEATURE_NAMES,
        "expected_feature_dim": EXPECTED_FEATURE_DIM,
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
            },
            "scaler_loaded": _scaler is not None
        }
    }

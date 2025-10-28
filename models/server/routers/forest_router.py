# routers/forest_router.py
"""
Forest cover prediction router for FastAPI.

Endpoints:
- POST  /forest/predict         -> single sample prediction
- POST  /forest/predict_batch   -> batch prediction
- GET   /forest/mappings        -> view categorical mappings
- POST  /forest/mappings        -> update mappings at runtime (in-memory)
- GET   /forest/info            -> model & feature info (paths + loaded status)

Drop your model files under the 'models/...' directories relative to the project root:
- models/scikit_models/random_forest_model.pkl
- models/xgboost_models/xgboost_forest_model.pkl
- models/tensorflow_models/forest_cover_model.keras
- models/tensorflow_quantized_model/forest_cover_model_quantized.tflite
"""
from typing import List, Union, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
import joblib
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path

CURRENT_FILE_DIR = Path(__file__).resolve().parent

router = APIRouter(prefix="/forest", tags=["forest"])

# ---------------------------
# Feature ordering
# ---------------------------
FEATURE_NAMES = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area",   # collapsed categorical
    "Soil_Type"          # collapsed categorical
]

EXPECTED_FEATURE_DIM = len(FEATURE_NAMES)  # 12

# ---------------------------
# Human-readable label mapping for Cover_Type (standard Covertype labels)
# ---------------------------
LABEL_MAP: Dict[int, str] = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

def _label_from_int(i: int) -> str:
    return LABEL_MAP.get(int(i), str(i))

# ---------------------------
# Default mappings (customize as needed)
# ---------------------------
WILDERNESS_MAP: Dict[str, int] = {
    "rawah wilderness area": 1,
    "rawah": 1,
    "neota wilderness area": 2,
    "neota": 2,
    "comanche peak wilderness area": 3,
    "comanche": 3,
    "cache la poudre wilderness area": 4,
    "cache la poudre": 4,
    "cache": 4
}

# Soil mapping: include soil_type1..soil_type40 plus full descriptions
SOIL_MAP: Dict[str, int] = {f"soil_type{i}": i for i in range(1, 41)}

SOIL_MAP.update({
    "cathedral family - rock outcrop complex, extremely stony.": 1,
    "vanet - ratake families complex, very stony.": 2,
    "haploborolis - rock outcrop complex, rubbly.": 3,
    "ratake family - rock outcrop complex, rubbly.": 4,
    "vanet family - rock outcrop complex complex, rubbly.": 5,
    "vanet - wetmore families - rock outcrop complex, stony.": 6,
    "gothic family.": 7,
    "supervisor - limber families complex.": 8,
    "troutville family, very stony.": 9,
    "bullwark - catamount families - rock outcrop complex, rubbly.": 10,
    "bullwark - catamount families - rock land complex, rubbly.": 11,
    "legault family - rock land complex, stony.": 12,
    "catamount family - rock land - bullwark family complex, rubbly.": 13,
    "pachic argiborolis - aquolis complex.": 14,
    "unspecified in the usfs soil and elu survey.": 15,
    "cryaquolis - cryoborolis complex.": 16,
    "gateview family - cryaquolis complex.": 17,
    "rogert family, very stony.": 18,
    "typic cryaquolis - borohemists complex.": 19,
    "typic cryaquepts - typic cryaquolls complex.": 20,
    "typic cryaquolls - leighcan family, till substratum complex.": 21,
    "leighcan family, till substratum, extremely bouldery.": 22,
    "leighcan family, till substratum - typic cryaquolls complex.": 23,
    "leighcan family, extremely stony.": 24,
    "leighcan family, warm, extremely stony.": 25,
    "granile - catamount families complex, very stony.": 26,
    "leighcan family, warm - rock outcrop complex, extremely stony.": 27,
    "leighcan family - rock outcrop complex, extremely stony.": 28,
    "como - legault families complex, extremely stony.": 29,
    "como family - rock land - legault family complex, extremely stony.": 30,
    "leighcan - catamount families complex, extremely stony.": 31,
    "catamount family - rock outcrop - leighcan family complex, extremely stony.": 32,
    "leighcan - catamount families - rock outcrop complex, extremely stony.": 33,
    "cryorthents - rock land complex, extremely stony.": 34,
    "cryumbrepts - rock outcrop - cryaquepts complex.": 35,
    "bross family - rock land - cryumbrepts complex, extremely stony.": 36,
    "rock outcrop - cryumbrepts - cryorthents complex, extremely stony.": 37,
    "leighcan - moran families - cryaquolls complex, extremely stony.": 38,
    "moran family - cryorthents - leighcan family complex, extremely stony.": 39,
    "moran family - cryorthents - rock land complex, extremely stony.": 40
})

# normalize map keys to lowercase (already lowercased strings but do safe normalization)
WILDERNESS_MAP = {k.strip().lower(): v for k, v in WILDERNESS_MAP.items()}
SOIL_MAP = {k.strip().lower(): v for k, v in SOIL_MAP.items()}

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
# Model file paths (absolute, computed from project root)
# ---------------------------
RF_PATH = CURRENT_FILE_DIR/'../../scikit_models/random_forest_model.pkl'
XGB_PATH =  CURRENT_FILE_DIR/'../../xgboost_models/xgboost_forest_model.pkl'
TF_PATH =  CURRENT_FILE_DIR/'../../tensorflow_models/forest_cover_model.keras'
TFLITE_PATH =  CURRENT_FILE_DIR/'../../tensorflow_quantized_model/forest_cover_model_quantized.tflite'

# We don't have a scaler for now
SCALER_PATH = None

# ---------------------------
# Model load helpers and status logs
# ---------------------------
LOAD_LOGS: Dict[str, str] = {}

def _load_joblib_safe(path: str):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        msg = f"file not found: {abs_path}"
        print(f"[models] {msg}")
        LOAD_LOGS[abs_path] = msg
        return None
    try:
        m = joblib.load(abs_path)
        msg = f"loaded (type={type(m)})"
        print(f"[models] {abs_path} -> {msg}")
        LOAD_LOGS[abs_path] = msg
        return m
    except Exception as e:
        msg = f"failed to load: {repr(e)}"
        print(f"[models] {abs_path} -> {msg}")
        LOAD_LOGS[abs_path] = msg
        return None

# ---------------------------
# Load models (safe)
# ---------------------------
_rf_model = _load_joblib_safe(RF_PATH)
_xgb_model = _load_joblib_safe(XGB_PATH)

_scaler = None  # explicit: no scaler right now

_tf_model = None
if os.path.exists(TF_PATH):
    try:
        _tf_model = tf.keras.models.load_model(TF_PATH)
        msg = f"loaded (tf keras model)"
        print(f"[models] {os.path.abspath(TF_PATH)} -> {msg}")
        LOAD_LOGS[os.path.abspath(TF_PATH)] = msg
    except Exception as e:
        msg = f"failed to load tf model: {repr(e)}"
        print(f"[models] {os.path.abspath(TF_PATH)} -> {msg}")
        LOAD_LOGS[os.path.abspath(TF_PATH)] = msg
        _tf_model = None
else:
    LOAD_LOGS[os.path.abspath(TF_PATH)] = f"file not found: {os.path.abspath(TF_PATH)}"
    print(f"[models] file not found: {os.path.abspath(TF_PATH)}")

_tflite_interpreter = None
_tflite_input_details = None
_tflite_output_details = None
if os.path.exists(TFLITE_PATH):
    try:
        _tflite_interpreter = tf.lite.Interpreter(model_path=os.path.abspath(TFLITE_PATH))
        _tflite_interpreter.allocate_tensors()
        _tflite_input_details = _tflite_interpreter.get_input_details()
        _tflite_output_details = _tflite_interpreter.get_output_details()
        msg = "loaded (tflite interpreter)"
        print(f"[models] {os.path.abspath(TFLITE_PATH)} -> {msg}")
        LOAD_LOGS[os.path.abspath(TFLITE_PATH)] = msg
    except Exception as e:
        msg = f"failed to load tflite: {repr(e)}"
        print(f"[models] {os.path.abspath(TFLITE_PATH)} -> {msg}")
        LOAD_LOGS[os.path.abspath(TFLITE_PATH)] = msg
        _tflite_interpreter = None
else:
    LOAD_LOGS[os.path.abspath(TFLITE_PATH)] = f"file not found: {os.path.abspath(TFLITE_PATH)}"
    print(f"[models] file not found: {os.path.abspath(TFLITE_PATH)}")

# ---------------------------
# Helpers for categorical mapping
# ---------------------------
def _map_wilderness(value: Union[str, int, float]) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s.isdigit():
            return int(s)
        if s in WILDERNESS_MAP:
            return WILDERNESS_MAP[s]
    raise HTTPException(status_code=422, detail=f"Unknown Wilderness_Area value: {value}")

def _map_soil(value: Union[str, int, float]) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s.isdigit():
            return int(s)
        if s in SOIL_MAP:
            return SOIL_MAP[s]
    raise HTTPException(status_code=422, detail=f"Unknown Soil_Type value: {value}")

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
            if name == "Wilderness_Area":
                mapped = _map_wilderness(val)
                arr.append(float(mapped))
            elif name == "Soil_Type":
                mapped = _map_soil(val)
                arr.append(float(mapped))
            else:
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
            fname = FEATURE_NAMES[idx]
            if fname == "Wilderness_Area":
                arr.append(float(_map_wilderness(val)))
            elif fname == "Soil_Type":
                arr.append(float(_map_soil(val)))
            else:
                try:
                    arr.append(float(val))
                except Exception:
                    raise HTTPException(status_code=422, detail=f"Feature {fname} must be numeric, got {val}")
        return np.array(arr, dtype=np.float32)
    else:
        raise HTTPException(status_code=422, detail="Sample must be a dict (feature->value) or a list of numbers/strings.")

# ---------------------------
# scaling helper (no scaler now)
# ---------------------------
def _apply_scaler(X: np.ndarray) -> np.ndarray:
    # no scaler currently — return raw
    return X

# ---------------------------
# Prediction helper wrappers
# ---------------------------
def _predict_rf(x: np.ndarray) -> Dict[str, Any]:
    if _rf_model is None:
        raise HTTPException(status_code=500, detail="RandomForest model not loaded.")
    # Use DataFrame with feature names to avoid sklearn feature-name mismatch warnings
    try:
        df = pd.DataFrame(x, columns=FEATURE_NAMES)
        pred = _rf_model.predict(df)
        probs = _rf_model.predict_proba(df).tolist() if hasattr(_rf_model, "predict_proba") else None
    except Exception:
        # fallback to numpy if DataFrame approach fails
        pred = _rf_model.predict(x)
        probs = _rf_model.predict_proba(x).tolist() if hasattr(_rf_model, "predict_proba") else None
    return {"predictions": pred.tolist(), "probabilities": probs}

def _predict_xgb(x: np.ndarray) -> Dict[str, Any]:
    if _xgb_model is None:
        raise HTTPException(status_code=500, detail="XGBoost model not loaded.")
    try:
        df = pd.DataFrame(x, columns=FEATURE_NAMES)
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
    classes = np.argmax(yp, axis=1) + 1
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
        pred_class = int(np.argmax(output_data, axis=1)[0]) + 1
        results.append(pred_class)
        probs.append(output_data.tolist()[0])
    return {"predictions": results, "probabilities": probs}

# ---------------------------
# Endpoints (predict, predict_batch)
# ---------------------------
@router.post("/predict", summary="Predict forest cover for a single sample")
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
            "label": _label_from_int(rf_pred),
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
            "label": _label_from_int(xgb_pred),
            "probabilities": xgb_out["probabilities"][0] if xgb_out["probabilities"] else None
        }
    except HTTPException as e:
        out["xgboost"] = {"error": str(e.detail)}

    # TensorFlow
    try:
        tf_in = arr_scaled
        tf_out = _predict_tf(tf_in)
        tf_pred = int(tf_out["predictions"][0])
        out["tensorflow"] = {
            "prediction": tf_pred,
            "label": _label_from_int(tf_pred),
            "probabilities": tf_out["probabilities"][0]
        }
    except HTTPException as e:
        out["tensorflow"] = {"error": str(e.detail)}

    # TFLite
    try:
        tflite_in = arr_scaled
        tflite_out = _predict_tflite(tflite_in)
        tflite_pred = int(tflite_out["predictions"][0])
        out["quantized"] = {
            "prediction": tflite_pred,
            "label": _label_from_int(tflite_pred),
            "probabilities": tflite_out["probabilities"][0]
        }
    except HTTPException as e:
        out["quantized"] = {"error": str(e.detail)}

    return out

@router.post("/predict_batch", summary="Predict forest cover for a batch of samples")
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
            "labels": [_label_from_int(x) for x in rf_preds],
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
            "labels": [_label_from_int(x) for x in xgb_preds],
            "probabilities": xgb_out["probabilities"]
        }
    except HTTPException as e:
        out["xgboost"] = {"error": str(e.detail)}

    # TF
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
# Mapping management endpoints
# ---------------------------
@router.get("/mappings", summary="Get current categorical mappings")
def get_mappings():
    return {
        "wilderness_map": WILDERNESS_MAP,
        "soil_map": SOIL_MAP
    }

@router.post("/mappings", summary="Update/extend categorical mappings")
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
# Info endpoint (paths + loaded status)
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

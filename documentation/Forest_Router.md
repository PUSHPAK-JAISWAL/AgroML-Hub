# Forest Router — API Documentation

Comprehensive reference for the `routers/forest_router.py` FastAPI router. Use this to test in Swagger, Postman, or to share with frontend/devs.

---

# Overview

Base path: **`/forest`**

Models exposed:

* `randomforest` — scikit-learn RandomForest (`.pkl`)
* `xgboost` — XGBoost (`.pkl`)
* `tensorflow` — Keras `.keras` model
* `quantized` — TFLite quantized model

All prediction endpoints return a JSON object with keys: `randomforest`, `xgboost`, `tensorflow`, `quantized`. Each model block contains `prediction` (integer 1–7), `label` (human string), and `probabilities` (list of floats for classes 1..7) when available.

---

# Feature names & ordering (INPUT MUST MATCH)

The router expects **12 features** in this order (also used for list-form inputs and DataFrame column names):

1. `Elevation`
2. `Aspect`
3. `Slope`
4. `Horizontal_Distance_To_Hydrology`
5. `Vertical_Distance_To_Hydrology`
6. `Horizontal_Distance_To_Roadways`
7. `Hillshade_9am`
8. `Hillshade_Noon`
9. `Hillshade_3pm`
10. `Horizontal_Distance_To_Fire_Points`
11. `Wilderness_Area`  — collapsed categorical (1..4). Accepts numeric, numeric-string, or friendly text (e.g. `"rawah"`, `"Rawah Wilderness Area"`)
12. `Soil_Type` — collapsed categorical (1..40). Accepts numeric, numeric-string, `"soil_type5"`, or full descriptive names (mapped in `SOIL_MAP`).

---

# Labels (human-readable mapping for class integers)

The `label` field maps class integers (1–7) to forest-cover names:

* `1` → **Spruce/Fir**
* `2` → **Lodgepole Pine**
* `3` → **Ponderosa Pine**
* `4` → **Cottonwood/Willow**
* `5` → **Aspen**
* `6` → **Douglas-fir**
* `7` → **Krummholz**

---

# Endpoints

---

## 1) POST `/forest/predict`

Predict a **single** sample.

**Request body (object)** — `sample` can be a dict (feature name → value) or a list (values ordered as FEATURE_NAMES).

Example (dict form):

```json
{
  "sample": {
    "Elevation": 2596,
    "Aspect": 51,
    "Slope": 3,
    "Horizontal_Distance_To_Hydrology": 258,
    "Vertical_Distance_To_Hydrology": 0,
    "Horizontal_Distance_To_Roadways": 510,
    "Hillshade_9am": 221,
    "Hillshade_Noon": 232,
    "Hillshade_3pm": 148,
    "Horizontal_Distance_To_Fire_Points": 606,
    "Wilderness_Area": "Rawah Wilderness Area",
    "Soil_Type": "Cathedral family - Rock outcrop complex, extremely stony."
  }
}
```

Example (list form):

```json
{
  "sample": [2596,51,3,258,0,510,221,232,148,606,"rawah","soil_type1"]
}
```

**Success response (200 OK)** — sample:

```json
{
  "randomforest": {
    "prediction": 2,
    "label": "Lodgepole Pine",
    "probabilities": [0.04,0.49,0.21,0,0.15,0.11,0]
  },
  "xgboost": {
    "prediction": 4,
    "label": "Cottonwood/Willow",
    "probabilities": [0.017,0.271,0.184,0.0006,0.522,0.00445,0.00017]
  },
  "tensorflow": {
    "prediction": 7,
    "label": "Krummholz",
    "probabilities": [0,0,0,0,0,0,1]
  },
  "quantized": {
    "prediction": 1,
    "label": "Spruce/Fir",
    "probabilities": [0.5,0.5,0,0,0,0,0]
  }
}
```

**Errors**

* `422 Unprocessable Entity` — missing features, wrong list length (list must be length 12), or unknown categorical mapping.
* `500 Internal Server Error` — model not loaded (returned within the model block as `{"error": "..."}'`).

---

## 2) POST `/forest/predict_batch`

Predict multiple samples in one call.

**Request body:**

```json
{
  "samples": [
    { ... },   // dict sample
    [ ... ]    // list sample (length 12)
  ]
}
```

**Success response (200 OK)** — sample:

```json
{
  "randomforest": {
    "predictions": [2, 5],
    "labels": ["Lodgepole Pine","Aspen"],
    "probabilities": [
      [0.04,0.49,0.21,0,0.15,0.11,0],
      [0.02,0.11,0.1,0.1,0.5,0.15,0.02]
    ]
  },
  "xgboost": { ... },
  "tensorflow": { ... },
  "quantized": { ... }
}
```

**Errors**: same as single predict (422, model errors returned per-model).

---

## 3) GET `/forest/mappings`

Return current in-memory categorical mappings.

**Response (200 OK)**:

```json
{
  "wilderness_map": {
    "rawah wilderness area": 1,
    "rawah": 1,
    "neota": 2,
    ...
  },
  "soil_map": {
    "soil_type1": 1,
    "cathedral family - rock outcrop complex, extremely stony.": 1,
    ...
  }
}
```

Note: mapping keys are lowercased in router; lookups are case-insensitive.

---

## 4) POST `/forest/mappings`

Update or extend mappings at runtime (in-memory only).

**Request body**:

```json
{
  "wilderness_map": {"my_area": 3},
  "soil_map": {"peaty": 12, "mysoil": 20}
}
```

**Response (200 OK)**:

```json
{
  "status": "ok",
  "updated": {
    "wilderness_added": [{"my_area": 3}],
    "soil_added": [{"peaty":12},{"mysoil":20}]
  }
}
```

> Reminder: Updates are in-memory unless you persist them (not done by default). They will be lost on server restart.

---

## 5) GET `/forest/info`

Model and feature info (useful for debugging).

**Response (200 OK)**:

```json
{
  "feature_names": [ ... ],
  "expected_feature_dim": 12,
  "models": {
    "randomforest": {"loaded": true, "path": "<abs path>", "log": "loaded (type=...)"},
    "xgboost": {...},
    "tensorflow": {...},
    "quantized_tflite": {...},
    "scaler_loaded": false
  }
}
```

If a `log` says `file not found` or `failed to load`, move the model file to that path or set `MODEL_ROOT` (if using the auto-detect version) or update the `RF_PATH`/`XGB_PATH` constants.

---

# Accepted value types for categorical features

* `Wilderness_Area`: integer (1..4) OR numeric string `"1"` OR short text `"rawah"` OR full `"Rawah Wilderness Area"`
* `Soil_Type`: integer (1..40) OR numeric string `"5"` OR `"soil_type5"` OR full description (see `SOIL_MAP` for exact texts).

If a textual value is not found, you’ll get `422` with message `Unknown Wilderness_Area value: ...` or `Unknown Soil_Type value: ...`. Use `POST /forest/mappings` to add new synonyms.

---

# Probabilities ordering

`probabilities` arrays are ordered for classes **1 → 7** (index 0→class1, index 1→class2, ...). Use `label` to read the human-readable class name.

---

# Troubleshooting

* **`X does not have valid feature names` warnings**: fixed by passing a `pandas.DataFrame` with `FEATURE_NAMES` to scikit/xgboost predictors. If you see warnings still, ensure your server environment has the same scikit-learn version used to save the model.
* **Models not loaded**: call `GET /forest/info` to inspect `path` and `log`. Move correct files to that path or adjust constants.
* **Inconsistent outputs between models**:

  * Check that all models were trained on the exact same preprocessing and feature order.
  * If TensorFlow was trained on scaled features and you do not supply the scaler, TF predictions will differ — save `StandardScaler` used in TF training and provide it to the router (or add upload endpoint).
* **Model load failures**: run a local Python test using the same virtualenv as uvicorn:

  ```python
  import joblib, numpy as np
  m = joblib.load("models/scikit_models/random_forest_model.pkl")
  print(m.predict(np.zeros((1,12))))
  ```

  If that fails, paste the exception into the console/logs.

---

# Example curl commands

Single predict (dict):

```bash
curl -X POST "http://127.0.0.1:8000/forest/predict" -H "Content-Type: application/json" -d '{
  "sample": {
    "Elevation": 2596, "Aspect": 51, "Slope": 3,
    "Horizontal_Distance_To_Hydrology": 258, "Vertical_Distance_To_Hydrology": 0,
    "Horizontal_Distance_To_Roadways": 510, "Hillshade_9am": 221,
    "Hillshade_Noon": 232, "Hillshade_3pm": 148,
    "Horizontal_Distance_To_Fire_Points": 606, "Wilderness_Area": "rawah", "Soil_Type": "1"
  }
}'
```

Batch predict:

```bash
curl -X POST "http://127.0.0.1:8000/forest/predict_batch" -H "Content-Type: application/json" -d '{
  "samples": [
    {"Elevation":2596,"Aspect":51,"Slope":3,"Horizontal_Distance_To_Hydrology":258,"Vertical_Distance_To_Hydrology":0,"Horizontal_Distance_To_Roadways":510,"Hillshade_9am":221,"Hillshade_Noon":232,"Hillshade_3pm":148,"Horizontal_Distance_To_Fire_Points":606,"Wilderness_Area":"rawah","Soil_Type":"soil_type1"},
    [2785,155,18,242,118,3090,238,238,122,622,"neota","soil_type5"]
  ]
}'
```

Get mappings:

```bash
curl "http://127.0.0.1:8000/forest/mappings"
```

Update mappings:

```bash
curl -X POST "http://127.0.0.1:8000/forest/mappings" -H "Content-Type: application/json" -d '{
  "wilderness_map": {"rawah_area": 1},
  "soil_map": {"cathedral": 1}
}'
```

Get info:

```bash
curl "http://127.0.0.1:8000/forest/info"
```

---

# Security & Production notes

* Validate and sanitize incoming requests in production. Rate-limit endpoints if public.
* Persist the `mappings` if you want them to survive restarts (write to `mappings.json` on update).
* For high throughput, serve scikit/xgboost models with dedicated model servers or set up a worker pool to avoid blocking.
* Add monitoring/logging to capture model drift and unusual inputs.

---

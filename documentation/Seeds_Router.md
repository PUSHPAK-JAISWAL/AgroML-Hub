# Seeds Router — API Documentation

Fast, copy-pasteable reference for the `routers/seeds_router.py` FastAPI router (wheat kernels / seeds dataset). Use this for Swagger, Postman, or to share with front-end/devs.

---

# Overview

**Base path:** `/seeds`

Exposed model backends (returned together in each prediction response):

* `randomforest` — scikit-learn RandomForest (`.pkl`)
* `xgboost` — XGBoost (either a plain model `.pkl` or a packaged joblib with `{"model": ..., "feature_columns": ...}`)
* `tensorflow` — Keras model (`.keras`)
* `quantized` — TFLite quantized model (`.tflite`)

Each model block in responses contains:

* `prediction` — integer label returned by the model (may be 0-based or 1-based depending on model)
* `label` — human-readable variety name (mapped when possible)
* `probabilities` — list of per-class probabilities (if model exposes them)

**No scaler is assumed** (router currently does not use or require a saved `scaler.pkl`). If you trained TF using a normalization layer inside the model, predictions will be consistent without an external scaler.

---

# Feature names & ordering (INPUT MUST MATCH)

The router expects **7 features** in this order (exact keys, case-sensitive) — use dictionary form or list form (ordered):

1. `Area`
2. `Perimeter`
3. `Compactness`
4. `length_of_kernel`
5. `width_of_kernel`
6. `asymetric_coef`
7. `length_of_kernel_groove`

Examples below show both dictionary and ordered-list formats. If sending dictionary, all keys **must** be present and spelled exactly as above. If sending list, it must contain exactly 7 numeric values in the order above.

---

# Class labels (human-readable mapping)

The router maps numeric predictions to these labels when possible:

* `1` → **Kama**
* `2` → **Rosa**
* `3` → **Canadian**

The code accepts both 1..3 and 0..2 encodings: if model returns 0..2 the router will map `0→1`, `1→2`, `2→3` for the label lookup. If a mapping is not found, the integer is returned as a string.

---

# Endpoints

## 1) `POST /seeds/predict`

Predict a **single** sample.

**Request body (object)** — `sample` may be:

* a **dict** with the 7 named features (keys must match exactly), e.g.:

```json
{
  "sample": {
    "Area": 15.26,
    "Perimeter": 14.84,
    "Compactness": 0.871,
    "length_of_kernel": 5.763,
    "width_of_kernel": 3.312,
    "asymetric_coef": 2.221,
    "length_of_kernel_groove": 5.22
  }
}
```

* or a **list** (ordered values) of length **7**, e.g.:

```json
{
  "sample": [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22]
}
```

**Success response (200)** — example:

```json
{
  "randomforest": {
    "prediction": 1,
    "label": "Kama",
    "probabilities": [0.8, 0.15, 0.05]
  },
  "xgboost": {
    "prediction": 0,
    "label": "Kama",
    "probabilities": [0.75, 0.2, 0.05]
  },
  "tensorflow": {
    "prediction": 0,
    "label": "Kama",
    "probabilities": [0.7, 0.25, 0.05]
  },
  "quantized": {
    "prediction": 0,
    "label": "Kama",
    "probabilities": [0.6, 0.35, 0.05]
  }
}
```

**Errors**

* `422 Unprocessable Entity` — missing feature (e.g. `"Missing feature: Area"`), wrong list length, or non-numeric value for a numeric feature.
* Model-specific errors are returned inside the corresponding model block as `{"error": "..."}` when a model isn't loaded.

---

## 2) `POST /seeds/predict_batch`

Predict multiple samples at once.

**Request body:**

```json
{
  "samples": [
    { ... },    // dict sample with named keys
    [ ... ]     // ordered-list sample
  ]
}
```

**Success response (200)** — example:

```json
{
  "randomforest": {
    "predictions": [1, 2],
    "labels": ["Kama", "Rosa"],
    "probabilities": [[0.8,0.15,0.05], [0.1,0.85,0.05]]
  },
  "xgboost": { ... },
  "tensorflow": { ... },
  "quantized": { ... }
}
```

**Errors**: same as single predict (422 for format issues). Model-specific errors returned per-model.

---

## 3) `GET /seeds/mappings`

Returns the current in-memory categorical mappings (kept for API parity; seeds dataset does not use these features).

**Response example:**

```json
{
  "wilderness_map": {},
  "soil_map": {}
}
```

---

## 4) `POST /seeds/mappings`

Update or extend mappings at runtime (in-memory). This is a no-op for the seeds dataset (kept for parity), but you can add entries and they’ll be stored in memory until server restart.

**Request example:**

```json
{
  "wilderness_map": {"some_alias": 1},
  "soil_map": {"some_soil": 2}
}
```

**Response:**

```json
{"status":"ok","updated":{"wilderness_added":[{"some_alias":1}],"soil_added":[{"some_soil":2}]}}
```

---

## 5) `GET /seeds/info`

Returns model & feature info useful for debugging.

**Response example:**

```json
{
  "feature_names": ["Area","Perimeter","Compactness","length_of_kernel","width_of_kernel","asymetric_coef","length_of_kernel_groove"],
  "expected_feature_dim": 7,
  "models": {
    "randomforest": {"loaded": true, "path": "<abs path>", "log": "loaded (type=...)" },
    "xgboost": {"loaded": true, "path": "<abs path>", "log": "loaded (type=...)" },
    "tensorflow": {"loaded": true, "path": "<abs path>", "log": "loaded (tf keras model)" },
    "quantized_tflite": {"loaded": true, "path": "<abs path>", "log": "loaded (tflite interpreter)" },
    "scaler_loaded": false
  }
}
```

---

# Probabilities ordering

`probabilities` arrays are ordered by class index as returned by the model. Common training patterns:

* If your labels were 1..3: `probabilities[0]` → class `1` (Kama), `[1]` → `2` (Rosa), `[2]` → `3` (Canadian).
* If your model outputs 0..2, router converts labels to human names using the +1 mapping for label translation: `0 -> Kama, 1 -> Rosa, 2 -> Canadian`.
* Always check the `label` field which maps the integer to a human name.

---

# Troubleshooting & common pitfalls

* **Missing feature errors**: keys are **case-sensitive**. Use `"Area"` not `"area"`. For dict input, include *all* feature names.
* **List input length**: list must be exactly 7 values.
* **Feature order mismatch**: if you send a list, ensure it follows the exact `FEATURE_NAMES` order.
* **Model not loaded**: call `GET /seeds/info` to inspect the `path` and `log`. Move model files to the expected paths or update the router constants.
* **Inconsistent outputs across models**:

  * Confirm each model was trained with the same preprocessing (e.g., normalization inside TF or external scaler). This router expects raw numeric values unless you add a scaler and adapt `_apply_scaler`.
  * Keras model in your training used a `Normalization` layer inside the model — that keeps TF consistent even without an external scaler.
* **XGBoost joblib package**: If you saved a package dict containing `model` and `feature_columns`, the router will detect and use `feature_columns` when building the DataFrame for predict. If not, it falls back to `FEATURE_NAMES`.

---

# Example curl commands (copy-paste)

Single predict (dict):

```bash
curl -X POST "http://127.0.0.1:8000/seeds/predict" \
 -H "Content-Type: application/json" \
 -d '{"sample":{"Area":15.26,"Perimeter":14.84,"Compactness":0.871,"length_of_kernel":5.763,"width_of_kernel":3.312,"asymetric_coef":2.221,"length_of_kernel_groove":5.22}}'
```

Single predict (list):

```bash
curl -X POST "http://127.0.0.1:8000/seeds/predict" \
 -H "Content-Type: application/json" \
 -d '{"sample":[15.26,14.84,0.871,5.763,3.312,2.221,5.22]}'
```

Batch predict:

```bash
curl -X POST "http://127.0.0.1:8000/seeds/predict_batch" \
 -H "Content-Type: application/json" \
 -d '{"samples":[{"Area":15.26,"Perimeter":14.84,"Compactness":0.871,"length_of_kernel":5.763,"width_of_kernel":3.312,"asymetric_coef":2.221,"length_of_kernel_groove":5.22}, [14.88,14.57,0.8811,5.554,3.333,1.018,4.956]]}'
```

Get info:

```bash
curl "http://127.0.0.1:8000/seeds/info"
```

Update mappings (no-op but allowed):

```bash
curl -X POST "http://127.0.0.1:8000/seeds/mappings" \
 -H "Content-Type: application/json" \
 -d '{"wilderness_map":{"alias":1},"soil_map":{"sandy":2}}'
```

---

# Production notes & suggested improvements

* Persist runtime mapping updates (save to `mappings.json`) if you want them across restarts. Currently they are in-memory only.
* Add `/models/reload` endpoint to reload models without restarting the process.
* If TF was trained on scaled inputs outside the model, supply a `scaler.pkl` and implement `_apply_scaler` to call it; I can add an endpoint to upload the scaler.
* Add request validation and rate limiting before exposing publicly.



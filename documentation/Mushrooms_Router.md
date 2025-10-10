# Mushroom Router — API Documentation

Fast, copy-pasteable reference for the `routers/mushroom_router.py` FastAPI router.
This router accepts either single-letter codes (the original dataset values) **or** full descriptive words (e.g. `"convex"`) for categorical fields — it will convert them to the numeric encodings used during training when possible.

---

# Overview

**Base path:** `/mushroom`

Returned model blocks (all returned together):

* `randomforest` — scikit-learn RandomForest (`.pkl`)
* `xgboost` — XGBoost (plain model or packaged joblib with `model`, `feature_columns`, `label_encoder`)
* `tensorflow` — Keras model (`.keras`)
* `quantized` — TFLite quantized model (`.tflite`)

Each model block contains:

* `prediction` — numeric label returned by the model (0/1 or package-specific)
* `label` — human-readable string (e.g. `edible` / `poisonous`) whenever the router can map it
* `probabilities` — per-class probability array (if model provides it)

No external scaler is required (TF models use an internal Normalization layer in your training script).

---

# Accepted input formats

For predictions the router accepts **either**:

1. A **dictionary** with all **22** feature keys present (case-sensitive, see `FEATURE_NAMES` below), OR
2. An **ordered list** of **22** values (must match the ordering in `FEATURE_NAMES`).

If you send words (e.g. `"convex"`) the router will attempt to map them to the encoding used during training. If you send the single-letter codes (e.g. `"x"`) the router will also handle them.

---

# Feature names & order (exact keys required for dict input)

```
[
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
```

---

# Allowed values (single-letter → full word)

You can send the single-letter code OR the full word (case-insensitive).
Below are the mappings accepted by the router (letter → description):

* **cap-shape:** bell=`b`, conical=`c`, convex=`x`, flat=`f`, knobbed=`k`, sunken=`s`
* **cap-surface:** fibrous=`f`, grooves=`g`, scaly=`y`, smooth=`s`
* **cap-color:** brown=`n`, buff=`b`, cinnamon=`c`, gray=`g`, green=`r`, pink=`p`, purple=`u`, red=`e`, white=`w`, yellow=`y`
* **bruises:** bruises=`t`, no=`f`
* **odor:** almond=`a`, anise=`l`, creosote=`c`, fishy=`y`, foul=`f`, musty=`m`, none=`n`, pungent=`p`, spicy=`s`
* **gill-attachment:** attached=`a`, descending=`d`, free=`f`, notched=`n`
* **gill-spacing:** close=`c`, crowded=`w`, distant=`d`
* **gill-size:** broad=`b`, narrow=`n`
* **gill-color:** black=`k`, brown=`n`, buff=`b`, chocolate=`h`, gray=`g`, green=`r`, orange=`o`, pink=`p`, purple=`u`, red=`e`, white=`w`, yellow=`y`
* **stalk-shape:** enlarging=`e`, tapering=`t`
* **stalk-root:** bulbous=`b`, club=`c`, cup=`u`, equal=`e`, rhizomorphs=`z`, rooted=`r`, missing=`?`
* **stalk-surface-above-ring:** fibrous=`f`, scaly=`y`, silky=`k`, smooth=`s`
* **stalk-surface-below-ring:** fibrous=`f`, scaly=`y`, silky=`k`, smooth=`s`
* **stalk-color-above-ring:** brown=`n`, buff=`b`, cinnamon=`c`, gray=`g`, orange=`o`, pink=`p`, red=`e`, white=`w`, yellow=`y`
* **stalk-color-below-ring:** brown=`n`, buff=`b`, cinnamon=`c`, gray=`g`, orange=`o`, pink=`p`, red=`e`, white=`w`, yellow=`y`
* **veil-type:** partial=`p`, universal=`u`
* **veil-color:** brown=`n`, orange=`o`, white=`w`, yellow=`y`
* **ring-number:** none=`n`, one=`o`, two=`t`
* **ring-type:** cobwebby=`c`, evanescent=`e`, flaring=`f`, large=`l`, none=`n`, pendant=`p`, sheathing=`s`, zone=`z`
* **spore-print-color:** black=`k`, brown=`n`, buff=`b`, chocolate=`h`, green=`r`, orange=`o`, purple=`u`, white=`w`, yellow=`y`
* **population:** abundant=`a`, clustered=`c`, numerous=`n`, scattered=`s`, several=`v`, solitary=`y`
* **habitat:** grasses=`g`, leaves=`l`, meadows=`m`, paths=`p`, urban=`u`, waste=`w`, woods=`d`

---

# Example payloads (you can paste these directly into Swagger)

### Single predict — using full descriptive words (dictionary)

```json
{
  "sample": {
    "cap-shape": "convex",
    "cap-surface": "smooth",
    "cap-color": "brown",
    "bruises": "bruises",
    "odor": "pungent",
    "gill-attachment": "free",
    "gill-spacing": "close",
    "gill-size": "narrow",
    "gill-color": "black",
    "stalk-shape": "enlarging",
    "stalk-root": "equal",
    "stalk-surface-above-ring": "smooth",
    "stalk-surface-below-ring": "smooth",
    "stalk-color-above-ring": "white",
    "stalk-color-below-ring": "white",
    "veil-type": "partial",
    "veil-color": "white",
    "ring-number": "one",
    "ring-type": "pendant",
    "spore-print-color": "black",
    "population": "scattered",
    "habitat": "urban"
  }
}
```

### Single predict — using single-letter codes (ordered list)

```json
{
  "sample": ["x","s","n","t","p","f","c","n","k","e","e","s","s","w","w","p","w","o","p","k","s","u"]
}
```

### Batch predict — mixed (two samples)

```json
{
  "samples": [
    {
      "cap-shape": "convex",
      "cap-surface": "smooth",
      "cap-color": "brown",
      "bruises": "bruises",
      "odor": "pungent",
      "gill-attachment": "free",
      "gill-spacing": "close",
      "gill-size": "narrow",
      "gill-color": "black",
      "stalk-shape": "enlarging",
      "stalk-root": "equal",
      "stalk-surface-above-ring": "smooth",
      "stalk-surface-below-ring": "smooth",
      "stalk-color-above-ring": "white",
      "stalk-color-below-ring": "white",
      "veil-type": "partial",
      "veil-color": "white",
      "ring-number": "one",
      "ring-type": "pendant",
      "spore-print-color": "black",
      "population": "scattered",
      "habitat": "urban"
    },
    ["p","x","s","n","t","p","f","c","n","k","e","e","s","s","w","w","p","w","o","p","k","s"]
  ]
}
```

---

# Responses

**Successful (200)** — sample response:

```json
{
  "randomforest": {
    "prediction": 1,
    "label": "poisonous",
    "probabilities": [0.12, 0.88]
  },
  "xgboost": {
    "prediction": 0,
    "label": "edible",
    "probabilities": [0.95, 0.05]
  },
  "tensorflow": {
    "prediction": 1,
    "label": "poisonous",
    "probabilities": [0.11, 0.89]
  },
  "quantized": {
    "prediction": 1,
    "label": "poisonous",
    "probabilities": [0.10, 0.90]
  }
}
```

* Interpretation: probabilities are usually `[P(class0), P(class1)]` — check `label` to see human mapping (router attempts to use saved label encoders to return `edible`/`poisonous`).

---

# Error responses

* `422 Unprocessable Entity` — missing feature (e.g. `"Missing feature: cap-shape"`), wrong list length, or non-encodable textual value.
* `500 Internal Server Error` — model-specific not-loaded errors will be returned inside the model block as `{"error": "RandomForest model not loaded."}` (router still returns the other model outputs if those are available).

---

# Helpful curl examples

Single (full words):

```bash
curl -X POST "http://127.0.0.1:8000/mushroom/predict" \
  -H "Content-Type: application/json" \
  -d '{"sample": {"cap-shape":"convex","cap-surface":"smooth","cap-color":"brown","bruises":"bruises","odor":"pungent","gill-attachment":"free","gill-spacing":"close","gill-size":"narrow","gill-color":"black","stalk-shape":"enlarging","stalk-root":"equal","stalk-surface-above-ring":"smooth","stalk-surface-below-ring":"smooth","stalk-color-above-ring":"white","stalk-color-below-ring":"white","veil-type":"partial","veil-color":"white","ring-number":"one","ring-type":"pendant","spore-print-color":"black","population":"scattered","habitat":"urban"}}'
```

Single (letters, ordered list):

```bash
curl -X POST "http://127.0.0.1:8000/mushroom/predict" \
  -H "Content-Type: application/json" \
  -d '{"sample":["x","s","n","t","p","f","c","n","k","e","e","s","s","w","w","p","w","o","p","k","s","u"]}'
```

Batch example:

```bash
curl -X POST "http://127.0.0.1:8000/mushroom/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{"samples":[{"cap-shape":"convex", "cap-surface":"smooth", "cap-color":"brown", "bruises":"bruises", "odor":"pungent", "gill-attachment":"free", "gill-spacing":"close", "gill-size":"narrow", "gill-color":"black", "stalk-shape":"enlarging","stalk-root":"equal","stalk-surface-above-ring":"smooth","stalk-surface-below-ring":"smooth","stalk-color-above-ring":"white","stalk-color-below-ring":"white","veil-type":"partial","veil-color":"white","ring-number":"one","ring-type":"pendant","spore-print-color":"black","population":"scattered","habitat":"urban"}, ["p","x","s","n","t","p","f","c","n","k","e","e","s","s","w","w","p","w","o","p","k","s"]] }'
```
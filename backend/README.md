# Agroml API — Backend README

This README documents the **Spring Boot backend** (API gateway) that forwards prediction requests to a FastAPI model server and handles authentication, user management, prediction history and notifications.
Give this document to frontend engineers or an LLM to generate UIs quickly.

> Base URLs (defaults used in development)

* Spring backend (this README): `http://localhost:8080`
* FastAPI model server (prediction microservice): `http://localhost:8000`
  (Spring forwards requests to `app.fastapi.base-url` configured in `application.yml`)

---

# Authentication (JWT)

## Endpoints

* `POST /auth/register` — register new user

    * Body: `{ "email": "<email>", "password": "<password>" }`
    * Response: `200 OK` JSON with created user (id, email, roles).

* `POST /auth/login` — login and get JWT

    * Body: `{ "email": "<email>", "password": "<password>" }`
    * Response: `200 OK` JSON:

      ```json
      { "token": "<jwt>", "tokenType": "Bearer" }
      ```

## Usage

Include the JWT for protected endpoints in the header:

```
Authorization: Bearer <jwt>
Content-Type: application/json
```

* Public endpoints: `/auth/**`, `/public/**`
* Admin endpoints require role `ROLE_ADMIN`: `/admin/**`
* Authenticated user endpoints require any authenticated user.

---

# Users (Admin CRUD)

> Base path: `/admin/users` — **ADMIN only**

* `GET /admin/users` — list all users (200)
* `GET /admin/users/{id}` — get a user by id (200 / 404)
* `POST /admin/users` — create user (body: User JSON, stored password will be hashed) (200)
* `PUT /admin/users/{id}` — update user (200)
* `DELETE /admin/users/{id}` — delete user (204)

User model (example JSON):

```json
{
  "id": "string",
  "email": "user@example.com",
  "password": "plaintext-on-create",
  "roles": ["ROLE_USER"]
}
```

> Note: In production, frontends should never display the `password` field; admin APIs return full user records. Consider mapping to DTO in UI.

---

# Predictions (gateway)

> Base path: `/predictions` — **authenticated users only**

This controller forwards requests to the FastAPI model server configured via `app.fastapi.base-url`. It also saves prediction history to MongoDB and triggers an email to the user.

## Endpoints

### `POST /predictions`

* Purpose: forward a prediction request to the FastAPI model router, save history, notify user.
* Body:

```json
{
  "modelPath": "/forest/predict",         // or "forest/predict" — gateway will normalize
  "sample": { ... }                       // payload forwarded as {"sample": ...}
}
```

* Behavior:

    * Builds URL: `${fastapi_base_url}${modelPath}`
    * Forwards POST with JSON `{ "sample": <sample> }`
    * Saves a `PredictionHistory` document:

      ```json
      {
        "userId": "<user-id>",
        "endpoint": "/forest/predict",
        "input": { "sample": ... },
        "response": { ... },   // exact response from FastAPI
        "createdAt": "2025-10-28T14:23:12Z"
      }
      ```
    * Sends an async email to the user's email with a short summary.
* Response: returns the JSON body from FastAPI (200) or forwarded error.

### `GET /predictions/history`

* Returns array of `PredictionHistory` documents for the logged-in user (ordered desc).

---

# Model-specific routers (available on FastAPI, forwarded by Spring)

Spring accepts `modelPath` and forwards to FastAPI. These are the primary model endpoints you will use.

## Forest (Covertype)

* FastAPI endpoints (examples):

    * `POST /forest/predict` — single sample prediction
    * `POST /forest/predict_batch` — batch prediction
    * `GET /forest/mappings` — current categorical mappings
    * `POST /forest/mappings` — update/extend mappings
    * `GET /forest/info` — model & feature info

### Forest Feature names (order & required keys)

```text
Elevation,
Aspect,
Slope,
Horizontal_Distance_To_Hydrology,
Vertical_Distance_To_Hydrology,
Horizontal_Distance_To_Roadways,
Hillshade_9am,
Hillshade_Noon,
Hillshade_3pm,
Horizontal_Distance_To_Fire_Points,
Wilderness_Area,
Soil_Type
```

* Expect 12 features for a single sample.
* Categorical fields: `Wilderness_Area`, `Soil_Type` — frontend may send numeric codes (1..N) or textual names; gateway maps textual names to numeric codes automatically.

### Forest: sample JSON (single)

```json
{
  "modelPath": "/forest/predict",
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
    "Horizontal_Distance_To_Fire_Points": 0,
    "Wilderness_Area": "rawah",         // or "Rawah Wilderness Area" or 1
    "Soil_Type": "soil_type5"           // or full descriptive text or 5
  }
}
```

### Forest: example response (gateway adds `label` for readability)

```json
{
  "randomforest": {
    "prediction": 2,
    "label": "Lodgepole Pine",
    "probabilities": [0.04,0.49,0.21,0,0.15,0.11,0]
  },
  "xgboost": { "prediction": 4, "label": "Cottonwood/Willow", "probabilities": [...] },
  "tensorflow": { "prediction": 7, "label": "Krummholz", "probabilities": [...] },
  "quantized": { "prediction": 1, "label": "Spruce/Fir", "probabilities": [...] }
}
```

**Label mapping** (cover type -> human label)
1: Spruce/Fir
2: Lodgepole Pine
3: Ponderosa Pine
4: Cottonwood/Willow
5: Aspen
6: Douglas-fir
7: Krummholz

---

## Seeds (Wheat kernels)

* FastAPI endpoint names you should use (example): `POST /seeds/predict`

### Required features (single sample)

```text
Area,
Perimeter,
Compactness,
length_of_kernel,
width_of_kernel,
asymetric_coef,
length_of_kernel_groove
```

### Seeds: sample JSON

```json
{
  "modelPath": "/seeds/predict",
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

### Seeds: response

Gateway returns the multi-model responses (randomforest, xgboost, tensorflow, quantized). Each sub-object contains:

* `prediction`: numeric class index (0/1/2 or label as trained)
* `label`: human-readable class (gateway tries to include label if model metadata present)
* `probabilities`: distribution (if available)

---

## Mushrooms

* FastAPI endpoint names: `POST /mushrooms/predict`

### Features (categorical)

All mushroom features are categorical. Frontend **may** send either:

* single-letter codes (e.g. `"cap-shape": "x"`) **or**
* full English names (e.g. `"cap-shape": "convex"`). The server maps full words to codes internally.

Mappings accepted (frontend may use words — server handles):

**cap-shape**

* bell = `b`, conical = `c`, convex = `x`, flat = `f`, knobbed = `k`, sunken = `s`

**cap-surface**

* fibrous = `f`, grooves = `g`, scaly = `y`, smooth = `s`

**cap-color**

* brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y

**bruises**

* bruises=t, no=f

**odor**

* almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s

**gill-attachment**

* attached=a, descending=d, free=f, notched=n

**gill-spacing**

* close=c, crowded=w, distant=d

**gill-size**

* broad=b, narrow=n

**gill-color**

* black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y

**stalk-shape**

* enlarging=e, tapering=t

**stalk-root**

* bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?

**stalk-surface-above-ring / below-ring**

* fibrous=f, scaly=y, silky=k, smooth=s

**stalk-color-above-ring / below-ring**

* brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y

**veil-type**

* partial=p, universal=u

**veil-color**

* brown=n, orange=o, white=w, yellow=y

**ring-number**

* none=n, one=o, two=t

**ring-type**

* cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z

**spore-print-color**

* black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y

**population**

* abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y

**habitat**

* grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d

### Mushrooms: sample JSON (use words)

```json
{
  "modelPath": "/mushrooms/predict",
  "sample": {
    "cap-shape": "convex",
    "cap-surface": "smooth",
    "cap-color": "brown",
    "bruises": "no",
    "odor": "almond",
    "gill-attachment": "free",
    "gill-spacing": "close",
    "gill-size": "broad",
    "gill-color": "brown",
    "stalk-shape": "enlarging",
    "stalk-root": "bulbous",
    "stalk-surface-above-ring": "smooth",
    "stalk-surface-below-ring": "smooth",
    "stalk-color-above-ring": "brown",
    "stalk-color-below-ring": "brown",
    "veil-type": "partial",
    "veil-color": "white",
    "ring-number": "one",
    "ring-type": "pendant",
    "spore-print-color": "brown",
    "population": "numerous",
    "habitat": "woods"
  }
}
```

---

# Forest mapping management (FastAPI router)

* `GET /forest/mappings` — get current `WILDERNESS_MAP` and `SOIL_MAP`.
* `POST /forest/mappings` — update maps. Example payload:

```json
{
  "wilderness_map": {"new_name": 2, "rawah": 1},
  "soil_map": {"peaty": 12}
}
```

Server lowercases keys and updates mapping in-memory.

---

# Error responses & validation

* Input validation errors from FastAPI are forwarded (often `422` with a `detail` message such as `Missing feature: Area`).
* Spring global exception handler returns structured JSON for validation errors (field-level messages) and `400`/`500` for other errors.
* If a requested model is not loaded, FastAPI/Spring may return `500` with message `RandomForest model not loaded.` (or similar).

---

# Quick cURL examples

### Register

```bash
curl -X POST "http://localhost:8080/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email":"user1@example.com","password":"mypassword123"}'
```

### Login

```bash
curl -X POST "http://localhost:8080/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"user1@example.com","password":"mypassword123"}'
```

### Forest predict (single)

```bash
curl -X POST "http://localhost:8080/predictions" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{ "modelPath": "/forest/predict", "sample": { "Elevation": 2596, "Aspect": 51, "Slope": 3, "Horizontal_Distance_To_Hydrology": 258, "Vertical_Distance_To_Hydrology": 0, "Horizontal_Distance_To_Roadways": 510, "Hillshade_9am": 221, "Hillshade_Noon": 232, "Hillshade_3pm": 148, "Horizontal_Distance_To_Fire_Points": 0, "Wilderness_Area": "rawah", "Soil_Type": "soil_type5" } }'
```

### Seeds predict

```bash
curl -X POST "http://localhost:8080/predictions" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{ "modelPath": "/seeds/predict", "sample": { "Area": 15.26, "Perimeter": 14.84, "Compactness": 0.871, "length_of_kernel": 5.763, "width_of_kernel": 3.312, "asymetric_coef": 2.221, "length_of_kernel_groove": 5.22 } }'
```

### Mushrooms predict

```bash
curl -X POST "http://localhost:8080/predictions" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{ "modelPath": "/mushrooms/predict", "sample": { "cap-shape":"convex","cap-surface":"smooth","cap-color":"brown","bruises":"no","odor":"almond","gill-attachment":"free","gill-spacing":"close","gill-size":"broad","gill-color":"brown","stalk-shape":"enlarging","stalk-root":"bulbous","stalk-surface-above-ring":"smooth","stalk-surface-below-ring":"smooth","stalk-color-above-ring":"brown","stalk-color-below-ring":"brown","veil-type":"partial","veil-color":"white","ring-number":"one","ring-type":"pendant","spore-print-color":"brown","population":"numerous","habitat":"woods" } }'
```

---
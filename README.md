# Real-Time ML Inference REST API & Capstone

A beginner-friendly machine-learning inference API built with FastAPI. The demo model classifies Iris flowers from four measurements and returns prediction probabilities.

## Project files

- `main.py` — FastAPI endpoints: `/`, `/health`, and `/predict`
- `train_model.py` — trains a demo Random Forest model and saves `model.pkl`
- `model.pkl` — generated model file (create it by running the training script)
- `requirements.txt` — pinned Python dependencies
- `Dockerfile` — container build and run instructions
- `test_main.py` — endpoint and validation tests

## Run locally (Python 3.11 recommended)

### 1. Create and activate a virtual environment

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies and train the demo model

```bash
python -m pip install -r requirements.txt
python train_model.py
```

### 3. Start the API

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` in your browser to test the API interactively.

## Example request

POST `/predict` with JSON:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

The response includes a numeric class prediction and probability for each model class. Class mapping: `0 = setosa`, `1 = versicolor`, `2 = virginica`.

## Run tests

```bash
pytest -q
```

## Run with Docker

Build:
```bash
docker build -t ml-inference-api .
```

Run:
```bash
docker run --rm -p 8000:8000 ml-inference-api
```

Then visit `http://127.0.0.1:8000/docs`.

## Architecture

Client JSON request → FastAPI validation → saved scikit-learn pipeline → prediction and class probabilities → JSON response.

## Important note

This is a demo model for demonstrating API packaging, testing, and Docker. If your capstone requires a specific trained model from an earlier task, replace the demo model and adapt the input schema to match that model. Do not claim it is your earlier model unless you actually replace it.

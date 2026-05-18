from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
from pathlib import Path

app = FastAPI(title="Used Car Price Prediction API")

# ✅ CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ PROJECT ROOT = D:\New folder
BASE_DIR = Path(__file__).resolve().parent.parent

# ✅ Model path
MODEL_PATH = BASE_DIR / "model" / "car_price_rf.pkl"

model = None


class CarInput(BaseModel):
    manufacturing_year: int
    km_driven: int
    brand: str
    model: str
    fuel_type: str
    transmission_type: str
    city: str
    bodytype: str
    number_of_owners: int


@app.on_event("startup")
def load_model():
    global model

    print("🔍 Looking for model at:", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("✅ Model loaded successfully")


@app.get("/")
def home():
    return {"message": "✅ Used Car Price Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict_price(data: CarInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(df)[0]
        return {"predicted_price": int(round(prediction))}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

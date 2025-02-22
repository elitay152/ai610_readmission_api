import os
import pickle
import numpy as np
import requests
import zipfile
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from encoding import encode_input
from contextlib import asynccontextmanager
import asyncio

# Amazon S3 Model URL (replace with your actual S3 URL)
S3_MODEL_URL = "https://ai610-readmissions-storage.s3.us-east-1.amazonaws.com/rfc_model.pkl"

# Model File Paths
model_path = "rfc_model.pkl"

# Global model variable
model = None

async def download_model():
    """Download the model from S3 if not already present."""
    if not os.path.exists(model_path):
        print(f"📥 Downloading model from {S3_MODEL_URL}...")
        response = requests.get(S3_MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print("✅ Model downloaded successfully!")
        else:
            print(f"❌ Failed to download model. Status code: {response.status_code}")
            return

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifecycle hook to start background model loading."""
    print("🚀 FastAPI started successfully!")
    asyncio.create_task(load_model())  # Start model loading in the background
    yield  # Allow FastAPI to start immediately

async def load_model():
    """Background task to load the model after FastAPI starts."""
    global model
    await download_model()  # Ensure model is downloaded first
    print("📡 Loading model into memory...")
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

class ModelInput(BaseModel):
    number_outpatient: int
    change: str
    gender: str
    age: str
    diabetesMed: str
    time_in_hospital: int
    num_medications: int
    number_diagnoses: int

@app.get("/")
def home():
    return {"message": "RandomForest API is running!", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: ModelInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again later.")

    try:
        encoded_categorical = encode_input(data)
        X_input = np.array([[data.number_outpatient] + encoded_categorical +
                            [data.time_in_hospital, data.num_medications, data.number_diagnoses]])
        probabilities = model.predict_proba(X_input)[0]
        predicted_class = model.predict(X_input)[0]
        confidence_score = round(probabilities[predicted_class] * 100, 2)
        display_category = "Yes" if predicted_class == 1 else "No"

        return {
            "prediction": display_category,
            "risk_score": f"{confidence_score}% probability of {display_category} readmission"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

PORT = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    print(f"🚀 Running FastAPI on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

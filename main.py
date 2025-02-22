import os
import pickle
import numpy as np
import uvicorn
import aiohttp
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from encoding import encode_input
from contextlib import asynccontextmanager

# Hugging Face model URL
HUGGING_FACE_MODEL_URL = "https://huggingface.co/emtay152/ai610-readmission-model/resolve/main/rfc_model.pkl"

# Model File Path
MODEL_PATH = "rfc_model.pkl"

# Global model variable
model = None

async def download_model():
    """Force re-download the model from Hugging Face every time."""
    print(f"📥 Downloading model from {HUGGING_FACE_MODEL_URL}...")
    async with aiohttp.ClientSession() as session:
        async with session.get(HUGGING_FACE_MODEL_URL) as response:
            if response.status == 200:
                with open(MODEL_PATH, "wb") as file:
                    file.write(await response.read())
                print("✅ Model downloaded successfully!")
            else:
                print(f"❌ Failed to download model. Status code: {response.status}")


async def load_model():
    """Load the model into memory after ensuring it's downloaded."""
    global model
    await download_model()  # Ensure model is downloaded first
    print("📡 Loading model into memory...")
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure model is fully loaded before starting FastAPI."""
    print("🚀 FastAPI is starting...")
    await load_model()  # Wait for model to load before serving requests
    print("✅ Model is ready, API is live!")
    yield  # Allow FastAPI to start only after model is loaded


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
    """Health check endpoint to verify if API is running and model is loaded."""
    return {
        "message": "RandomForest API is running!",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: ModelInput):
    """Prediction endpoint that returns readmission probability."""
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


# Run FastAPI server
PORT = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    print(f"🚀 Running FastAPI on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

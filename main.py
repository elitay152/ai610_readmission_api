import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from encoding import encode_input
import gdown
import zipfile
import uvicorn

# Google Drive model file ID
file_id = '1bo361_iBxWL421SDDk_NaN7Cq5izxmat'
zip_path = "rfc_model.zip"
model_path = "rfc_model.pkl"

# Check if the model file exists and is a valid size
def is_valid_model(file_path, min_size_mb=5):
    """Check if the model file exists and is at least `min_size_mb` MB."""
    return os.path.exists(file_path) and os.path.getsize(file_path) > (min_size_mb * 1024 * 1024)

# Initialize FastAPI app
app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    print("🚀 Starting FastAPI server...")

    try:
        if not is_valid_model(model_path):
            print("📥 Model file missing or incomplete. Downloading compressed file from Google Drive...")
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            gdown.download(url, zip_path, quiet=False)

            print("📦 Extracting model...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(".")
            print("✅ Model extracted successfully!")

        # Load the model
        print("📡 Loading model into memory...")
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

# Define request structure
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
    return {"message": "RandomForest API is running!"}

@app.post("/predict")
def predict(data: ModelInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

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

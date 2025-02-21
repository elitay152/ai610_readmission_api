# 🏥 Diabetes Readmission Prediction API

This is a **FastAPI-based Machine Learning API** that predicts whether a patient is likely to be readmitted to the hospital based on various health-related factors. The model used is a **Random Forest Classifier**.

---

## 🚀 Features
- 🏥 **Predicts hospital readmission probability**
- 🎛 **Encodes categorical inputs automatically**
- 🔥 **Deployable via Railway or other cloud services**
- ⚡ **FastAPI provides an interactive API documentation**

---

## 🔍 API Endpoints
#### 📌 Home Route
- GET /
- Returns a simple message to confirm the API is running.
#### 📌 Predict Readmission
- POST /predict

---

#### Request Body Example(JSON):
{
  "number_outpatient": 2,
  "change": "Ch",
  "gender": "Male",
  "age": "30-60",
  "diabetesMed": "Yes",
  "time_in_hospital": 5,
  "num_medications": 10,
  "number_diagnoses": 3
}

#### Response Example (JSON):
{
  "prediction": "Yes",
  "risk_score": "87.6% probability of Yes readmission"
}

# TrueHire Prediction API

This is the FastAPI backend for the TrueHire ML prediction system. It provides an endpoint to predict whether a job posting is potentially fraudulent using a pre-trained Linear SVM model.

## Setup & Installation

Ensure you are in the project root directory where `backend/` and `ml/` are located.

1. **Install requirements:**
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Start the API Server:**
   ```bash
   uvicorn backend.app:app --reload
   ```
   The API will be available at: http://127.0.0.1:8000

## API Endpoints

### `GET /health`
Returns the status of the server and the currently loaded model.

**Example Response:**
```json
{
  "status": "ok",
  "model": "linear_svm_tuned"
}
```

### `POST /predict`
Predicts if a job posting is fraudulent.

**Request Format:**
Pass a JSON object with the job posting fields. Only `title` is strictly required, but passing all available fields improves prediction accuracy.

**Example Request:**
```json
{
  "title": "Software Engineer",
  "location": "US, CA, San Francisco",
  "department": "Engineering",
  "salary_range": "100000-120000",
  "company_profile": "We are a great company.",
  "description": "You will write code.",
  "requirements": "Python, FastAPI",
  "benefits": "Health insurance",
  "telecommuting": 1,
  "has_company_logo": 1,
  "has_questions": 0,
  "employment_type": "Full-time",
  "required_experience": "Mid-Senior level",
  "required_education": "Bachelor's Degree",
  "industry": "Internet",
  "function": "Engineering"
}
```

**Example Response:**
```json
{
  "prediction": "Likely Legitimate",
  "fraudulent": 0,
  "decision_score": -1.92487641864461
}
```

> **Note on `decision_score`**: The returned `decision_score` is the raw output from the Linear Support Vector Machine's decision function. It is **NOT** a calibrated probability (i.e. it does not range exclusively from 0 to 1, and "0.82" does not mean an 82% chance of fraud). It represents the confidence and distance from the decision boundary. Positive values or values above our threshold (0.20) indicate higher risk of fraud.

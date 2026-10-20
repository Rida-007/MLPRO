import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from backend.app import app

def run_tests():
    with TestClient(app) as client:
        print("Testing GET /health")
        response = client.get("/health")
        assert response.status_code == 200
        print("GET /health passed:", response.json())
        
        print("\nTesting POST /predict with full valid data")
        valid_data = {
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
        response = client.post("/predict", json=valid_data)
        assert response.status_code == 200
        resp_json = response.json()
        assert "prediction" in resp_json
        assert "fraudulent" in resp_json
        assert "decision_score" in resp_json
        print("POST /predict (full) passed:", resp_json)
        
        print("\nTesting POST /predict with missing optional fields")
        missing_data = {
            "title": "Warehouse Worker"
        }
        response = client.post("/predict", json=missing_data)
        assert response.status_code == 200
        resp_json = response.json()
        assert "prediction" in resp_json
        assert "fraudulent" in resp_json
        assert "decision_score" in resp_json
        print("POST /predict (missing optional fields) passed:", resp_json)

        print("\nTesting POST /predict with invalid input")
        invalid_data = {
            "department": "Engineering" # missing title
        }
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422 # Unprocessable Entity
        print("POST /predict (invalid input) passed with 422 Unprocessable Entity")

if __name__ == "__main__":
    run_tests()

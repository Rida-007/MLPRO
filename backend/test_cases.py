import requests
import json

url = "http://127.0.0.1:8000/predict"

cases = [
    {
        "name": "Test 1: Detailed, realistic legitimate job posting",
        "payload": {
            "title": "Senior Marketing Manager",
            "location": "US, NY, New York",
            "department": "Marketing",
            "salary_range": "110000-140000",
            "company_profile": "We are a well-established retail corporation with 50+ years of history.",
            "description": "We are looking for a marketing manager to lead our digital campaigns. You will work closely with the design team...",
            "requirements": "5+ years experience, Bachelor's degree in Marketing.",
            "benefits": "401k, health, dental, paid time off.",
            "telecommuting": 0,
            "has_company_logo": 1,
            "has_questions": 1,
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "industry": "Retail",
            "function": "Marketing"
        }
    },
    {
        "name": "Test 2: Sparse posting with minimal company info",
        "payload": {
            "title": "Customer Service Rep",
            "location": "US, FL, Miami",
            "description": "Call center rep needed. Apply now.",
            "telecommuting": 0,
            "has_company_logo": 0,
            "has_questions": 0
        }
    },
    {
        "name": "Test 3: Suspicious recruitment/payment language",
        "payload": {
            "title": "Data Entry Work From Home - Make $500/day",
            "location": "US",
            "description": "Earn $500 a day easily by typing from home! No experience required. We will send you a check to buy equipment. Email us your bank details to get started immediately.",
            "requirements": "None",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
            "employment_type": "Part-time"
        }
    },
    {
        "name": "Test 4: Realistic technical job with complete info",
        "payload": {
            "title": "DevOps Engineer",
            "location": "US, TX, Austin",
            "department": "Engineering",
            "salary_range": "130000-160000",
            "company_profile": "Tech Innovators Inc is a cloud solutions provider helping enterprise clients scale.",
            "description": "Join our infrastructure team to manage Kubernetes clusters and CI/CD pipelines.",
            "requirements": "Experience with AWS, Kubernetes, Terraform, and Python.",
            "benefits": "Comprehensive health, stock options, unlimited PTO.",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "industry": "Information Technology and Services",
            "function": "Information Technology"
        }
    },
    {
        "name": "Test 5: Deliberately incomplete posting",
        "payload": {
            "title": "Assistant"
        }
    }
]

print("=== TrueHire Backend E2E Test Cases ===")
for case in cases:
    print(f"\n{case['name']}")
    try:
        response = requests.post(url, json=case['payload'])
        print(f"HTTP Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Prediction: {data['prediction']}")
            print(f"Decision score: {data['decision_score']:.4f}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to connect: {e}")

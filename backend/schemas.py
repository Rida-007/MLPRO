from pydantic import BaseModel, Field
from typing import Optional

class JobPosting(BaseModel):
    title: str = Field(..., description="Job title")
    location: Optional[str] = None
    department: Optional[str] = None
    salary_range: Optional[str] = None
    company_profile: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[str] = None
    benefits: Optional[str] = None
    telecommuting: int = 0
    has_company_logo: int = 0
    has_questions: int = 0
    employment_type: Optional[str] = None
    required_experience: Optional[str] = None
    required_education: Optional[str] = None
    industry: Optional[str] = None
    function: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: str
    fraudulent: int
    decision_score: float

class HealthResponse(BaseModel):
    status: str
    model: str

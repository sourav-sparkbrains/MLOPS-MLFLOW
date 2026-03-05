import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Annotated
from fastapi import FastAPI, Depends,Header, HTTPException

from utils.inference import do_inference

load_dotenv()

app = FastAPI()

#---------------Pydantic models------------------
class EmployeeInput(BaseModel):
    years_experience: int
    team_size: int
    salary_usd_k: int
    ai_tools_used_per_day: int
    hours_with_ai_assistance_daily: int
    ai_replaces_my_tasks_pct: int
    weekly_ai_upskilling_hrs: int
    productivity_score: int
    burnout_score: int
    job_satisfaction_1_5: int
    job_role: str
    country: str
    industry: str
    remote_work_type: str
    primary_ai_tool: str
    ai_adoption_stage: str
    education_level: str
    company_size: str
    fear_of_ai_replacement: str

#------------Helper Function ----------------
API_KEY = os.getenv("API_KEY")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")


#------------Endpoints--------------
@app.get("/")
def check_health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: EmployeeInput,api_key:Annotated[str,Depends(verify_api_key)]):
    return do_inference(request)

@app.get("/model-info")
def model_info():
    return {
        "model_name": "BestAttritionModel",
        "model_alias": "production",
        "model_type": "XGBoost",
        "input_features": {
            "numeric": [
                "years_experience", "team_size", "salary_usd_k",
                "ai_tools_used_per_day", "hours_with_ai_assistance_daily",
                "ai_replaces_my_tasks_pct", "weekly_ai_upskilling_hrs",
                "productivity_score", "burnout_score", "job_satisfaction_1_5"
            ],
            "categorical_ohe": [
                "job_role", "country", "industry",
                "remote_work_type", "primary_ai_tool", "ai_adoption_stage"
            ],
            "categorical_ordinal": [
                "education_level", "company_size", "fear_of_ai_replacement"
            ]
        },
        "output_classes": ["Low", "Medium", "High"],
        "description": "Predicts employee attrition risk based on AI workplace factors"
    }

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    feature_1: float = Field(..., description="First feature value")
    feature_2: float = Field(..., description="Second feature value")
    feature_3: float = Field(..., description="Third feature value")
    feature_4: float = Field(..., description="Fourth feature value")
    
    class Config:
        json_schema_extra = {
            "example": {
                "feature_1": 5.1,
                "feature_2": 3.5,
                "feature_3": 1.4,
                "feature_4": 0.2
            }
        }


class PredictionOutput(BaseModel):
    prediction: int
    confidence: float
    model_version: str = "1.0"


class HealthCheck(BaseModel):
    status: str
    model_loaded: bool

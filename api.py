from fastapi import FastAPI, HTTPException
from schemas import PredictionInput, PredictionOutput, HealthCheck
from model import model_handler


app = FastAPI(
    title="ML Model API",
    description="REST API for ML model predictions",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "ML Model API",
        "endpoints": ["/predict", "/health", "/docs"]
    }


@app.get("/health", response_model=HealthCheck)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_handler.is_loaded()
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if not model_handler.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = [
            input_data.feature_1,
            input_data.feature_2,
            input_data.feature_3,
            input_data.feature_4
        ]
        
        prediction, confidence = model_handler.predict(features)
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "model_version": "1.0"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model/info")
def model_info():
    if not model_handler.is_loaded():
        return {"status": "Model not loaded"}
    
    return {
        "model_type": type(model_handler.model).__name__,
        "status": "loaded"
    }

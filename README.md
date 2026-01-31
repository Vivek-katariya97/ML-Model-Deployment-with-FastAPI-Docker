# ML Model Deployment with FastAPI & Docker

REST API for serving ML model predictions using FastAPI and Docker.

## Project Structure

```
├── api.py              # FastAPI application
├── model.py            # Model loading and inference
├── schemas.py          # Pydantic models for validation
├── train_model.py      # Create sample model
├── test_api.py         # API testing script
├── Dockerfile          # Docker configuration
├── requirements.txt
└── README.md
```

## Features

- **REST API**: FastAPI endpoints for predictions
- **Input Validation**: Pydantic schemas
- **Model Loading**: Reliable model inference
- **Docker**: Containerized deployment
- **Error Handling**: Comprehensive error responses
- **Testing**: API endpoint tests

## Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

Create sample model:
```bash
python train_model.py
```

Run API locally:
```bash
uvicorn api:app --reload
```

Test endpoints:
```bash
python test_api.py
```

Visit API docs: `http://localhost:8000/docs`

## Docker Deployment

Build image:
```bash
docker build -t ml-api .
```

Run container:
```bash
docker run -p 8000:8000 ml-api
```

## API Endpoints

### GET /
Root endpoint with API info

### GET /health
Health check and model status

### POST /predict
Make predictions
```json
{
  "feature_1": 5.1,
  "feature_2": 3.5,
  "feature_3": 1.4,
  "feature_4": 0.2
}
```

### GET /model/info
Model information

## Testing

The `test_api.py` script tests:
- Root endpoint
- Health check
- Predictions
- Input validation
- Error handling

## Author

Built to demonstrate ML model deployment with modern tools.

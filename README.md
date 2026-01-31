# ML Model Deployment with FastAPI & Docker

This project focuses on deploying a trained machine learning model as a production-style REST API using FastAPI, with Docker for consistent and portable deployment.

## What this project does
- Serves ML model predictions through RESTful API endpoints  
- Loads trained models efficiently for inference  
- Validates incoming requests to prevent invalid inputs  
- Handles errors gracefully and returns clear responses  

## Deployment
The application is containerized using Docker to ensure:
- Environment consistency across systems  
- Easy setup and reproducible deployments  
- Simplified local testing and future cloud deployment  

## Why I built this
The goal of this project was to move beyond notebook-based ML and understand how trained models are exposed as services in real-world systems.

## Tech stack
- Python  
- FastAPI  
- scikit-learn  
- Docker  

## How to run locally

### Using Docker
```bash
docker build -t ml-fastapi-app .
docker run -p 8000:8000 ml-fastapi-app

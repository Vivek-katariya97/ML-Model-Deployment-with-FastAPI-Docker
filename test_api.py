import requests
import json


BASE_URL = "http://localhost:8000"


def test_root():
    print("Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_health():
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_predict():
    print("Testing prediction endpoint...")
    
    data = {
        "feature_1": 5.1,
        "feature_2": 3.5,
        "feature_3": 1.4,
        "feature_4": 0.2
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Input: {data}")
    print(f"Response: {response.json()}\n")


def test_invalid_input():
    print("Testing invalid input...")
    
    data = {
        "feature_1": "invalid",
        "feature_2": 3.5
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_model_info():
    print("Testing model info endpoint...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


if __name__ == '__main__':
    print("="*60)
    print("API Testing Suite")
    print("="*60 + "\n")
    
    try:
        test_root()
        test_health()
        test_model_info()
        test_predict()
        test_invalid_input()
        
        print("="*60)
        print("All tests completed!")
        print("="*60)
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API")
        print("Make sure the API is running on http://localhost:8000")

import pickle
import numpy as np
from pathlib import Path


class ModelHandler:
    
    def __init__(self, model_path='model.pkl'):
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Model loaded from {self.model_path}")
            else:
                print(f"Model file not found at {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict(self, features):
        if self.model is None:
            raise ValueError("Model not loaded")
        
        input_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(input_array)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(input_array)
            confidence = float(np.max(probabilities))
        else:
            confidence = 1.0
        
        return int(prediction[0]), confidence
    
    def is_loaded(self):
        return self.model is not None


model_handler = ModelHandler()

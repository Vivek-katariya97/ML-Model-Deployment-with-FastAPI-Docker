from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle


def create_sample_model():
    np.random.seed(42)
    
    X_train = np.random.rand(100, 4) * 10
    y_train = np.random.randint(0, 3, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Sample model created and saved to model.pkl")


if __name__ == '__main__':
    create_sample_model()

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_disease_model():
    print("Training Crop Disease Model...")
    # Features: Temperature, Humidity, Soil Moisture
    np.random.seed(42)
    n_samples = 500
    
    # Generate mock data
    X = np.random.rand(n_samples, 3)
    X[:, 0] = X[:, 0] * 20 + 15 # Temp: 15-35 C
    X[:, 1] = X[:, 1] * 50 + 30 # Humidity: 30-80 %
    X[:, 2] = X[:, 2] * 40 + 20 # Soil Moisture: 20-60 %
    
    # Mock logic: high humidity + high temp -> Rust (1), low moisture -> Blight (2), else Healthy (0)
    y = []
    for i in range(n_samples):
        if X[i, 1] > 65 and X[i, 0] > 28:
            y.append("गेहूँ का रस्ट (Wheat Rust)")
        elif X[i, 2] < 30:
            y.append("झुलसा रोग (Blight)")
        else:
            y.append("स्वस्थ (Healthy)")
            
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    model.fit(X, y)
    
    joblib.dump(model, os.path.join(MODELS_DIR, 'disease_model.joblib'))
    print("Disease model saved.")

def train_yield_model():
    print("Training Crop Yield Model...")
    # Features: Area (ha), Rainfall (mm), Temperature (C), Fertilizer (kg/ha)
    np.random.seed(42)
    n_samples = 500
    
    X = np.random.rand(n_samples, 4)
    X[:, 0] = X[:, 0] * 10 + 1 # Area: 1-11 ha
    X[:, 1] = X[:, 1] * 400 + 100 # Rainfall: 100-500 mm
    X[:, 2] = X[:, 2] * 20 + 20 # Temp: 20-40 C
    X[:, 3] = X[:, 3] * 150 + 50 # Fertilizer: 50-200 kg/ha
    
    # Mock logic: Yield = Area * (Rainfall/100) * (Fertilizer/100) * random_factor
    y = X[:, 0] * (X[:, 1] / 100.0) * (X[:, 3] / 100.0) * (np.random.rand(n_samples) * 0.5 + 0.8)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    model.fit(X, y)
    
    joblib.dump(model, os.path.join(MODELS_DIR, 'yield_model.joblib'))
    print("Yield model saved.")

def train_recommend_model():
    print("Training Crop Recommendation Model...")
    # Features: N, P, K, pH, Rainfall
    np.random.seed(42)
    n_samples = 500
    
    X = np.random.rand(n_samples, 5)
    X[:, 0] = X[:, 0] * 100 + 20 # N: 20-120
    X[:, 1] = X[:, 1] * 60 + 10  # P: 10-70
    X[:, 2] = X[:, 2] * 60 + 10  # K: 10-70
    X[:, 3] = X[:, 3] * 3 + 5.5  # pH: 5.5-8.5
    X[:, 4] = X[:, 4] * 500 + 100 # Rainfall: 100-600 mm
    
    y = []
    for i in range(n_samples):
        if X[i, 4] > 400:
            y.append("कपास (Cotton)")
        elif X[i, 4] < 200:
            y.append("बाजरा (Bajra)")
        elif X[i, 0] > 70:
            y.append("गेहूँ (Wheat)")
        else:
            y.append("सरसों (Mustard)")
            
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    model.fit(X, y)
    
    joblib.dump(model, os.path.join(MODELS_DIR, 'recommend_model.joblib'))
    print("Recommend model saved.")

if __name__ == "__main__":
    train_disease_model()
    train_yield_model()
    train_recommend_model()
    print("All models trained successfully!")

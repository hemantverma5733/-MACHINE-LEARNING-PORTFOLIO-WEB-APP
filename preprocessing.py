import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def preprocess_churn_data(df):
    """
    Preprocess the churn dataset: encode categoricals, scale numericals.
    """
    df = df.copy()
    
    # Handle missing values (if any)
    df = df.fillna(df.median(numeric_only=True))
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical variables
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Save encoders for later use during prediction
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'churn_label_encoders.joblib'))

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'churn_scaler.joblib'))

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def preprocess_housing_data(df):
    """
    Preprocess the housing dataset.
    """
    df = df.copy()
    
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'housing_scaler.joblib'))

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

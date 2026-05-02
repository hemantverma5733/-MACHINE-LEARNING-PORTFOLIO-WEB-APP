import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from data_loader import load_housing_data
from preprocessing import preprocess_housing_data
from evaluation import evaluate_regression
from utils import save_model

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def train_and_evaluate():
    print("Loading Housing Data...")
    df = load_housing_data()
    
    print("Preprocessing Housing Data...")
    X_train, X_test, y_train, y_test, _ = preprocess_housing_data(df)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
    }
    
    best_model_name = None
    best_model = None
    best_r2 = -float('inf')
    
    results = {}
    
    print("Training Regression Models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_regression(y_test, y_pred)
        results[name] = metrics
        
        print(f"\n{name} Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        if metrics['R2 Score'] > best_r2:
            best_r2 = metrics['R2 Score']
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with R2 Score: {best_r2:.4f}")
    
    # Save the best model
    model_path = os.path.join(MODELS_DIR, 'best_housing_model.joblib')
    save_model(best_model, model_path)
    
if __name__ == "__main__":
    train_and_evaluate()

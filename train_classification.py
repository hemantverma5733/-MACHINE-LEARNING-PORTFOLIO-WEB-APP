import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from data_loader import load_churn_data
from preprocessing import preprocess_churn_data
from evaluation import evaluate_classification
from utils import save_model

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def train_and_evaluate():
    print("Loading Churn Data...")
    df = load_churn_data()
    
    print("Preprocessing Churn Data...")
    X_train, X_test, y_train, y_test, _ = preprocess_churn_data(df)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    
    best_model_name = None
    best_model = None
    best_f1 = -1
    
    results = {}
    
    print("Training Classification Models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_classification(y_test, y_pred)
        results[name] = metrics
        
        print(f"\n{name} Results:")
        for k, v in metrics.items():
            if k != 'Confusion Matrix':
                print(f"{k}: {v:.4f}")
        
        if metrics['F1 Score'] > best_f1:
            best_f1 = metrics['F1 Score']
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with F1 Score: {best_f1:.4f}")
    
    # Save the best model
    model_path = os.path.join(MODELS_DIR, 'best_churn_model.joblib')
    save_model(best_model, model_path)
    
if __name__ == "__main__":
    train_and_evaluate()

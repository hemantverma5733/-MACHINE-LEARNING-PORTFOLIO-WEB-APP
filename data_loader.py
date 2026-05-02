import os
import ssl
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing, make_classification

# Fix for macOS SSL certificate verification error when downloading datasets
ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def load_churn_data(save_csv=True):
    """
    Generates a synthetic Customer Churn dataset.
    """
    X, y = make_classification(
        n_samples=2000, 
        n_features=10, 
        n_informative=6, 
        n_redundant=2, 
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    
    # Make some features look more like real-world categorical/numeric data
    df['MonthlyCharges'] = np.abs(df['feature_0']) * 20 + 20
    df['Tenure'] = np.abs(df['feature_1']).astype(int) * 12
    df['Gender'] = np.where(df['feature_2'] > 0, 'Male', 'Female')
    df['Contract'] = pd.cut(df['feature_3'], bins=3, labels=['Month-to-month', 'One year', 'Two year'])
    df['InternetService'] = pd.cut(df['feature_4'], bins=3, labels=['DSL', 'Fiber optic', 'No'])
    
    # Drop the synthetic features that we replaced
    df = df.drop(columns=[f'feature_{i}' for i in range(5)])
    df['Churn'] = y
    
    if save_csv:
        df.to_csv(os.path.join(DATA_DIR, 'churn_data.csv'), index=False)
        
    return df

def load_housing_data(save_csv=True):
    """
    Loads California Housing dataset.
    """
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    if save_csv:
        df.to_csv(os.path.join(DATA_DIR, 'housing_data.csv'), index=False)
        
    return df

def load_movie_data(save_csv=True):
    """
    Generates a synthetic Movie dataset for content-based recommendation.
    """
    movies = [
        {"item_id": 1, "title": "The Quantum Paradox", "genre": "Sci-Fi", "description": "A brilliant physicist discovers a way to travel between parallel universes, but must stop a catastrophic collapse of reality."},
        {"item_id": 2, "title": "Laughing Matters", "genre": "Comedy", "description": "A group of stand-up comedians embark on a cross-country road trip, facing hilarious challenges and discovering the true meaning of friendship."},
        {"item_id": 3, "title": "Shadows of the Past", "genre": "Thriller", "description": "A detective haunted by a past case must solve a new string of murders that seem eerily similar to the ones he never solved."},
        {"item_id": 4, "title": "Love in Paris", "genre": "Romance", "description": "Two strangers meet by chance in the romantic city of Paris and spend a magical week together before going their separate ways."},
        {"item_id": 5, "title": "Galactic Warriors", "genre": "Sci-Fi Action", "description": "An elite team of soldiers must defend the galaxy from an ancient alien threat that has awakened from hibernation."},
        {"item_id": 6, "title": "The Haunted Manor", "genre": "Horror", "description": "A family moves into an old, secluded mansion, only to realize that they are not alone and the house holds dark secrets."},
        {"item_id": 7, "title": "Tech Startup Story", "genre": "Drama", "description": "A group of ambitious college dropouts build a revolutionary tech company, but success brings unexpected personal and professional challenges."},
        {"item_id": 8, "title": "Space Explorers", "genre": "Sci-Fi Documentary", "description": "A visually stunning journey through the cosmos, exploring the latest discoveries in astronomy and space travel."},
        {"item_id": 9, "title": "The Last Stand", "genre": "Action", "description": "A retired soldier is pulled back into action to save his kidnapped daughter from a ruthless international crime syndicate."},
        {"item_id": 10, "title": "Culinary Delights", "genre": "Documentary", "description": "An exploration of world cuisines, following renowned chefs as they create their signature dishes and share their passion for food."}
    ]
    
    df = pd.DataFrame(movies)
    
    if save_csv:
        df.to_csv(os.path.join(DATA_DIR, 'movie_data.csv'), index=False)
        
    return df

if __name__ == "__main__":
    print("Generating and saving datasets...")
    load_churn_data()
    load_housing_data()
    load_movie_data()
    print("Datasets saved to data/ directory.")

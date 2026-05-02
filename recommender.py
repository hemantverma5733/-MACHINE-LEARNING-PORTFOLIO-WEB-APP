import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_movie_data
import joblib
from utils import save_model

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def build_recommender():
    """
    Builds a Content-Based Recommendation System using TF-IDF and Cosine Similarity.
    """
    print("Loading Movie Data...")
    df = load_movie_data()
    
    # Combine features for content-based filtering
    # In a real scenario, you'd use NLTK/spaCy for lemmatization/stopword removal here.
    # TfidfVectorizer handles basic stopword removal and lowercasing natively.
    df['combined_features'] = df['genre'] + " " + df['description']
    
    print("Building TF-IDF Matrix...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    
    print("Calculating Similarity Matrix...")
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Save the vectorizer, similarity matrix, and movie dataset
    print("Saving Recommender Artifacts...")
    save_model(vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
    save_model(similarity_matrix, os.path.join(MODELS_DIR, 'similarity_matrix.joblib'))
    
    # Save the dataframe directly for easy access in the app
    df.to_pickle(os.path.join(MODELS_DIR, 'movies_df.pkl'))
    
    print("Recommendation engine built successfully!")
    
def get_recommendations(title, df, similarity_matrix, top_n=3):
    """
    Returns top_n movie recommendations based on content similarity.
    """
    try:
        idx = df[df['title'] == title].index[0]
    except IndexError:
        return []
    
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Skip the first one because it is the movie itself
    sim_scores = sim_scores[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

if __name__ == "__main__":
    build_recommender()

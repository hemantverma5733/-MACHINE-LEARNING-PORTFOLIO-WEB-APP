# Machine Learning Portfolio Project

A complete, production-style Machine Learning Web Application built with Python, Scikit-learn, and Streamlit. This project showcases three different machine learning paradigms in a modular, easy-to-use interface.

## Features

1. **Customer Churn Prediction (Classification)**
   - Uses synthetic telecom data to predict whether a customer will churn.
   - Evaluates models (Logistic Regression, Random Forest, Decision Tree) using Accuracy, Precision, Recall, and F1 Score.
   - Serves the best-performing model to predict churn probability for new inputs.

2. **House Price Prediction (Regression)**
   - Uses the California Housing Dataset to estimate median house values.
   - Compares Linear Regression, Decision Tree, and Random Forest Regressors.
   - Evaluates performance using MAE, MSE, RMSE, and R2 Score.

3. **Movie Recommender (Recommendation System)**
   - A content-based recommendation engine built using TF-IDF vectorization and Cosine Similarity on movie titles, genres, and descriptions.

## Technology Stack

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **NLP (Text Processing)**: Scikit-learn (TfidfVectorizer)
- **Web UI**: Streamlit
- **Model Serialization**: Joblib

## Project Structure

```
ml_portfolio_project/
├── app.py                     # Main Streamlit UI application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── data/                      # Generated dataset CSVs
├── models/                    # Serialized models and scalers (.joblib, .pkl)
├── src/
│   ├── data_loader.py         # Data acquisition and synthetic data generation
│   ├── preprocessing.py       # Feature scaling and encoding
│   ├── train_classification.py# Churn model training pipeline
│   ├── train_regression.py    # Housing model training pipeline
│   ├── recommender.py         # Recommendation system builder
│   ├── evaluation.py          # Metrics calculator
│   └── utils.py               # Helper functions (saving/loading models)
```

## Setup Instructions

1. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Data and Train Models**
   Run the following scripts to generate data, train the models, and save the artifacts in the `models/` folder:
   ```bash
   PYTHONPATH=src python src/train_classification.py
   PYTHONPATH=src python src/train_regression.py
   PYTHONPATH=src python src/recommender.py
   ```

4. **Run the Streamlit App**
   Launch the user interface locally:
   ```bash
   streamlit run app.py
   ```

## Future Improvements

- Integrate deep learning models using PyTorch or TensorFlow.
- Implement collaborative filtering for the recommendation engine using user-interaction data.
- Deploy the Streamlit app to Streamlit Community Cloud, Heroku, or AWS.

import logging
import os
import joblib

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

def save_model(model, model_path):
    """
    Saves a trained model to the specified path.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """
    Loads a trained model from the specified path.
    """
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"No model found at {model_path}")

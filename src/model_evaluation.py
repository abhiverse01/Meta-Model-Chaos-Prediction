# src/model_evaluation.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from utils import setup_logger

# Set up logging
logger = setup_logger('model_evaluation', 'logs/model_evaluation.log')

def evaluate_meta_model():
    # Load test data and meta-model
    df = pd.read_csv('data/titanic_preprocessed.csv')
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the meta-model
    meta_model = joblib.load('models/meta_model.pkl')
    
    # Generate meta-features for test data
    models = load_models()
    X_meta_test = generate_meta_features(models, X_test)
    
    # Predict and evaluate
    y_meta_pred = meta_model.predict(X_meta_test)
    accuracy = accuracy_score(y_test, y_meta_pred)
    logger.info(f"Meta-model accuracy on test data: {accuracy:.4f}")
    
    # Detailed classification report and confusion matrix
    logger.info("\n" + classification_report(y_test, y_meta_pred))
    logger.info("\n" + str(confusion_matrix(y_test, y_meta_pred)))

if __name__ == "__main__":
    evaluate_meta_model()

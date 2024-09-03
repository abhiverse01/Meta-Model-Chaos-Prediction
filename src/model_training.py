import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import joblib
import logging
from utils import setup_logger

# Set up logging
logger = setup_logger('model_training', 'logs/model_training.log')

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    logger.info("Trained Logistic Regression model.")
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    logger.info("Trained Decision Tree model.")
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    logger.info("Trained Random Forest model.")
    return model

def train_svm(X_train, y_train):
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Trained SVM model.")
    return model

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    logger.info("Trained XGBoost model.")
    return model

def save_model(model, model_name):
    """
    Save the trained model to the models directory.
    
    :param model: Trained model object.
    :param model_name: Name of the model file.
    """
    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved {model_name} model to {model_path}.")

def train_and_save_models():
    # Load preprocessed data
    df = pd.read_csv('data/titanic_preprocessed.csv')
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    # Train models
    logreg_model = train_logistic_regression(X, y)
    dt_model = train_decision_tree(X, y)
    rf_model = train_random_forest(X, y)
    svm_model = train_svm(X, y)
    xgb_model = train_xgboost(X, y)
    
    # Save models
    save_model(logreg_model, 'logistic_regression')
    save_model(dt_model, 'decision_tree')
    save_model(rf_model, 'random_forest')
    save_model(svm_model, 'svm')
    save_model(xgb_model, 'xgboost')

if __name__ == "__main__":
    train_and_save_models()

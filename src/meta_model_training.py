# src/meta_model_training.py

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from utils import setup_logger

# Set up logging
logger = setup_logger('meta_model_training', 'logs/meta_model_training.log')

def load_models():
    """
    Load the trained base models.
    
    :return: Dictionary of models
    """
    models = {
        'logreg': joblib.load('models/logistic_regression.pkl'),
        'dt': joblib.load('models/decision_tree.pkl'),
        'rf': joblib.load('models/random_forest.pkl'),
        'svm': joblib.load('models/svm.pkl'),
        'xgb': joblib.load('models/xgboost.pkl'),
        'ann': tf.keras.models.load_model('models/ann_model.h5'),
        'lstm': tf.keras.models.load_model('models/lstm_model.h5')
    }
    logger.info("Loaded all base models.")
    return models

def generate_meta_features(models, X):
    """
    Generate meta-features from the predictions of base models.
    
    :param models: Dictionary of models.
    :param X: Feature set.
    :return: DataFrame of meta-features.
    """
    meta_features = np.column_stack((
        models['logreg'].predict(X), models['logreg'].predict_proba(X)[:, 1],
        models['dt'].predict(X), models['dt'].predict_proba(X)[:, 1],
        models['rf'].predict(X), models['rf'].predict_proba(X)[:, 1],
        models['svm'].predict(X), models['svm'].predict_proba(X)[:, 1],
        models['xgb'].predict(X), models['xgb'].predict_proba(X)[:, 1],
        models['ann'].predict(X).flatten(), (models['ann'].predict(X) > 0.5).astype(int).flatten(),
        models['lstm'].predict(X.values.reshape((X.shape[0], X.shape[1], 1))).flatten(),
        (models['lstm'].predict(X.values.reshape((X.shape[0], X.shape[1], 1))) > 0.5).astype(int).flatten()
    ))

    meta_df = pd.DataFrame(meta_features, columns=[
        'logreg_pred', 'logreg_prob', 
        'dt_pred', 'dt_prob',
        'rf_pred', 'rf_prob',
        'svm_pred', 'svm_prob',
        'xgb_pred', 'xgb_prob',
        'ann_pred', 'ann_prob',
        'lstm_pred', 'lstm_prob'
    ])
    
    logger.info("Generated meta-features for meta-model training.")
    return meta_df

def train_meta_model(X_meta, y_meta):
    """
    Train the meta-model using the meta-features.
    
    :param X_meta: Meta-features DataFrame.
    :param y_meta: True labels.
    :return: Trained meta-model.
    """
    meta_model = RandomForestClassifier(random_state=42)
    meta_model.fit(X_meta, y_meta)
    joblib.dump(meta_model, 'models/meta_model.pkl')
    logger.info("Trained and saved the meta-model.")
    return meta_model

if __name__ == "__main__":
    # Load preprocessed data and split it
    df = pd.read_csv('data/titanic_preprocessed.csv')
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load models and generate meta-features
    models = load_models()
    X_meta_train = generate_meta_features(models, X_train)
    X_meta_test = generate_meta_features(models, X_test)
    
    # Train the meta-model
    meta_model = train_meta_model(X_meta_train, y_train)
    
    # Evaluate the meta-model
    y_meta_pred = meta_model.predict(X_meta_test)
    accuracy = (y_meta_pred == y_test).mean()
    logger.info(f"Meta-model accuracy on test data: {accuracy:.4f}")

# src/dl_model_training.py

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
import joblib
import logging
from utils import setup_logger

# Set up logging
logger = setup_logger('deep_learning_training', 'logs/deep_learning_training.log')

def build_ann(input_shape):
    model = Sequential([
        Dense(128, input_dim=input_shape, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logger.info("Built ANN model.")
    return model

def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, input_shape=(input_shape, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(50),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logger.info("Built LSTM model.")
    return model

def train_ann(X_train, y_train):
    ann_model = build_ann(X_train.shape[1])
    ann_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    ann_model.save("models/ann_model.h5")
    logger.info("Trained and saved ANN model.")

def train_lstm(X_train, y_train):
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    lstm_model = build_lstm(X_train.shape[1])
    lstm_model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2)
    lstm_model.save("models/lstm_model.h5")
    logger.info("Trained and saved LSTM model.")

def train_and_save_deep_learning_models():
    # Load preprocessed data
    df = pd.read_csv('data/titanic_preprocessed.csv')
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    # Train ANN
    train_ann(X, y)
    
    # Train LSTM
    train_lstm(X, y)

if __name__ == "__main__":
    train_and_save_deep_learning_models()

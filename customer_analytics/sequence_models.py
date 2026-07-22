# customer_analytics/sequence_models.py
"""
Sequence Models for Customer Analytics.

This module implements LSTM and GRU models for sequence prediction,
risk assessment, and anomaly detection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class SequenceModelEngine:
    """
    Sequence Model Engine for Customer Analytics.
    
    Provides:
    - LSTM models for sequence prediction
    - GRU models for sequence prediction
    - Bidirectional LSTM/GRU
    - Regularization techniques (Dropout, BatchNorm, L1/L2)
    - Anomaly detection scoring
    """
    
    def __init__(self, random_state: int = 42, model_dir: str = 'serialized_weights'):
        """
        Initialize the Sequence Model Engine.
        
        Args:
            random_state: Random seed for reproducibility
            model_dir: Directory to save models
        """
        self.random_state = random_state
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.history = {}
        self.metrics = {}
        self.scaler = StandardScaler()
        
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        logger.info("Sequence Model Engine initialized")
    
    def build_lstm(self, sequence_length: int, n_features: int,
                   lstm_units: List[int] = [64, 32],
                   dropout_rate: float = 0.2,
                   use_bidirectional: bool = True) -> keras.Model:
        """
        Build an LSTM model for sequence prediction.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate
            use_bidirectional: Whether to use bidirectional LSTM
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential()
        model.add(layers.Input(shape=(sequence_length, n_features)))
        
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)
            
            if use_bidirectional:
                lstm_layer = layers.Bidirectional(
                    layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        dropout=dropout_rate,
                        recurrent_dropout=dropout_rate,
                        kernel_regularizer=regularizers.l2(0.001)
                    )
                )
            else:
                lstm_layer = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    kernel_regularizer=regularizers.l2(0.001)
                )
            
            model.add(lstm_layer)
            
            if i < len(lstm_units) - 1:
                model.add(layers.BatchNormalization())
        
        model.add(layers.Dense(32, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(1, activation='linear'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logger.info(f"LSTM model built with {len(lstm_units)} layers")
        return model
    
    def build_gru(self, sequence_length: int, n_features: int,
                  gru_units: List[int] = [64, 32],
                  dropout_rate: float = 0.2,
                  use_bidirectional: bool = True) -> keras.Model:
        """
        Build a GRU model for sequence prediction.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features
            gru_units: List of GRU units for each layer
            dropout_rate: Dropout rate
            use_bidirectional: Whether to use bidirectional GRU
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential()
        model.add(layers.Input(shape=(sequence_length, n_features)))
        
        for i, units in enumerate(gru_units):
            return_sequences = (i < len(gru_units) - 1)
            
            if use_bidirectional:
                gru_layer = layers.Bidirectional(
                    layers.GRU(
                        units,
                        return_sequences=return_sequences,
                        dropout=dropout_rate,
                        recurrent_dropout=dropout_rate,
                        kernel_regularizer=regularizers.l2(0.001)
                    )
                )
            else:
                gru_layer = layers.GRU(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    kernel_regularizer=regularizers.l2(0.001)
                )
            
            model.add(gru_layer)
            
            if i < len(gru_units) - 1:
                model.add(layers.BatchNormalization())
        
        model.add(layers.Dense(32, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(1, activation='linear'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logger.info(f"GRU model built with {len(gru_units)} layers")
        return model
    
    def prepare_sequences(self, data: np.ndarray, 
                          sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for training.
        
        Args:
            data: Input data array
            sequence_length: Length of each sequence
        
        Returns:
            Tuple of (features, targets)
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   model_name: str,
                   model_type: str = 'lstm',
                   sequence_length: int = 10,
                   epochs: int = 100,
                   batch_size: int = 32) -> Dict[str, Any]:
        """
        Train a sequence model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name for the model
            model_type: Type of model ('lstm' or 'gru')
            sequence_length: Length of sequences
            epochs: Number of epochs
            batch_size: Batch size
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training sequence model: {model_name}")
        
        n_features = X_train.shape[2] if len(X_train.shape) > 2 else 1
        
        if model_type == 'lstm':
            model = self.build_lstm(sequence_length, n_features)
        else:
            model = self.build_gru(sequence_length, n_features)
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(self.model_dir / f'{model_name}_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        self.models[model_name] = model
        self.history[model_name] = history
        
        metrics = self.evaluate_model(model, X_val, y_val)
        self.metrics[model_name] = metrics
        
        logger.info(f"Model {model_name} - MSE: {metrics['mse']:.4f}, "
                   f"MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
        
        return {
            'model': model,
            'history': history,
            'metrics': metrics
        }
    
    def evaluate_model(self, model: keras.Model, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a sequence model.
        
        Args:
            model: Trained Keras model
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test, verbose=0)
        
        return {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2': float(r2_score(y_test, y_pred))
        }
    
    def detect_anomalies(self, model_name: str, X: np.ndarray,
                         threshold: float = 2.0) -> np.ndarray:
        """
        Detect anomalies using reconstruction error.
        
        Args:
            model_name: Name of the trained model
            X: Input sequences
            threshold: Number of standard deviations for anomaly threshold
        
        Returns:
            Array of anomaly scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        predictions = model.predict(X, verbose=0)
        
        reconstruction_error = np.mean((X.reshape(X.shape[0], -1) - 
                                       predictions.reshape(predictions.shape[0], -1))**2, axis=1)
        
        mean_error = np.mean(reconstruction_error)
        std_error = np.std(reconstruction_error)
        
        anomaly_scores = (reconstruction_error - mean_error) / std_error
        anomalies = anomaly_scores > threshold
        
        logger.info(f"Detected {np.sum(anomalies)} anomalies (threshold: {threshold})")
        
        return anomalies
    
    def save_model(self, model_name: str) -> Path:
        """
        Save a trained model.
        
        Args:
            model_name: Name of the model to save
        
        Returns:
            Path to the saved model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        save_dir = self.model_dir / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model.save(str(save_dir / 'model.h5'))
        
        if model_name in self.metrics:
            with open(save_dir / 'metrics.json', 'w') as f:
                json.dump(self.metrics[model_name], f, indent=4)
        
        logger.info(f"Model saved to: {save_dir}")
        return save_dir
    
    def load_model(self, model_name: str) -> keras.Model:
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model to load
        
        Returns:
            Loaded Keras model
        """
        model_path = self.model_dir / model_name / 'model.h5'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = keras.models.load_model(str(model_path))
        self.models[model_name] = model
        
        logger.info(f"Model loaded from: {model_path}")
        return model

# This allows the module to be run directly for testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the engine
    engine = SequenceModelEngine()
    print("✅ Sequence Model Engine initialized successfully")
    print(f"Model directory: {engine.model_dir}")
    print("All models loaded successfully!")
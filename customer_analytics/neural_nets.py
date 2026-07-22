# customer_analytics/neural_nets.py
"""
Neural Networks for Customer Analytics.

This module provides implementations of neural network models for
customer classification, risk assessment, and conversion prediction.
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

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class NeuralNetworkEngine:
    """
    Neural Network Engine for Customer Analytics.
    
    Provides:
    - Multi-layer Perceptron (MLP) for classification
    - Deep Neural Networks with regularization
    - Activation function comparison (ReLU, Sigmoid, Tanh)
    - Optimizer comparison (Adam, SGD, RMSprop)
    - Training with early stopping and callbacks
    """
    
    def __init__(self, random_state: int = 42, model_dir: str = 'serialized_weights'):
        """
        Initialize the Neural Network Engine.
        
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
        self.label_encoder = LabelEncoder()
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        logger.info("Neural Network Engine initialized")
        logger.info(f"Model directory: {self.model_dir}")
    
    def build_mlp(self, input_shape: int, 
                  hidden_layers: List[int] = [128, 64, 32],
                  activation: str = 'relu',
                  dropout_rate: float = 0.3,
                  output_activation: str = 'sigmoid') -> keras.Model:
        """
        Build a Multi-Layer Perceptron (MLP) model.
        
        Args:
            input_shape: Number of input features
            hidden_layers: List of neurons in each hidden layer
            activation: Activation function ('relu', 'sigmoid', 'tanh')
            dropout_rate: Dropout rate for regularization
            output_activation: Activation for output layer
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential()
        model.add(layers.Input(shape=(input_shape,)))
        
        # Hidden layers
        for i, neurons in enumerate(hidden_layers):
            model.add(layers.Dense(
                neurons, 
                activation=activation,
                kernel_regularizer=regularizers.l2(0.001),
                name=f'dense_{i+1}'
            ))
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(1, activation=output_activation, name='output'))
        
        # Compile with default Adam optimizer
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        logger.info(f"MLP model built with {len(hidden_layers)} hidden layers")
        logger.info(f"Activation: {activation}, Dropout: {dropout_rate}")
        
        return model
    
    def build_deep_nn(self, input_shape: int, 
                      architecture: str = 'wide_deep') -> keras.Model:
        """
        Build a deep neural network with specialized architecture.
        
        Args:
            input_shape: Number of input features
            architecture: Architecture type ('wide_deep' or 'deep_only')
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=(input_shape,))
        
        if architecture == 'wide_deep':
            # Wide branch - simpler representation
            wide = layers.Dense(64, activation='relu')(inputs)
            wide = layers.Dropout(0.2)(wide)
            
            # Deep branch - complex representation
            deep = layers.Dense(256, activation='relu')(inputs)
            deep = layers.BatchNormalization()(deep)
            deep = layers.Dropout(0.3)(deep)
            deep = layers.Dense(128, activation='relu')(deep)
            deep = layers.BatchNormalization()(deep)
            deep = layers.Dropout(0.3)(deep)
            deep = layers.Dense(64, activation='relu')(deep)
            
            # Concatenate branches
            merged = layers.Concatenate()([wide, deep])
            
        else:
            # Deep only architecture
            x = layers.Dense(256, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(64, activation='relu')(x)
            merged = x
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(merged)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        logger.info(f"Deep NN model built with {architecture} architecture")
        return model
    
    def get_activation_comparison_models(self, input_shape: int) -> Dict[str, keras.Model]:
        """
        Create models with different activation functions for comparison.
        
        Args:
            input_shape: Number of input features
        
        Returns:
            Dictionary of models with different activations
        """
        activations = ['relu', 'sigmoid', 'tanh']
        models_dict = {}
        
        for activation in activations:
            model = self.build_mlp(
                input_shape, 
                hidden_layers=[128, 64, 32],
                activation=activation
            )
            models_dict[activation.capitalize()] = model
        
        logger.info(f"Created activation comparison models: {list(models_dict.keys())}")
        return models_dict
    
    def get_optimizer_comparison_models(self, input_shape: int) -> Dict[str, keras.Model]:
        """
        Create models with different optimizers for comparison.
        
        Args:
            input_shape: Number of input features
        
        Returns:
            Dictionary of models with different optimizers
        """
        optimizers = {
            'Adam': Adam(learning_rate=0.001),
            'SGD': SGD(learning_rate=0.01, momentum=0.9),
            'RMSprop': RMSprop(learning_rate=0.001)
        }
        
        models_dict = {}
        
        for opt_name, optimizer in optimizers.items():
            model = self.build_mlp(input_shape, [128, 64, 32], 'relu')
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
            models_dict[opt_name] = model
        
        logger.info(f"Created optimizer comparison models: {list(models_dict.keys())}")
        return models_dict
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   model_name: str, 
                   model: Optional[keras.Model] = None,
                   activation: str = 'relu',
                   epochs: int = 50, 
                   batch_size: int = 32,
                   class_weight: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train a neural network model with callbacks.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name for the model
            model: Pre-built model (optional)
            activation: Activation function
            epochs: Number of epochs
            batch_size: Batch size
            class_weight: Class weights for imbalance
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training model: {model_name}")
        
        # Build model if not provided
        if model is None:
            model = self.build_mlp(X_train.shape[1], [128, 64, 32], activation)
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(self.model_dir / f'{model_name}_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weight,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Store model and history
        self.models[model_name] = model
        self.history[model_name] = history
        
        # Evaluate
        metrics = self.evaluate_model(model, X_val, y_val)
        self.metrics[model_name] = metrics
        
        logger.info(f"Model {model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        
        return {
            'model': model,
            'history': history,
            'metrics': metrics
        }
    
    def evaluate_model(self, model: keras.Model, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained Keras model
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Predict
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        return metrics
    
    def predict(self, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model
            X: Input features
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        probabilities = model.predict(X, verbose=0)
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def save_model(self, model_name: str) -> Path:
        """
        Save a trained model and associated objects.
        
        Args:
            model_name: Name of the model to save
        
        Returns:
            Path to the saved model directory
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        save_dir = self.model_dir / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save(str(save_dir / 'model.h5'))
        
        # Save metrics
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
    
    def plot_training_history(self, model_name: str, 
                             save_path: Optional[Path] = None) -> None:
        """
        Plot training history for a model.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if model_name not in self.history:
            raise ValueError(f"History for model {model_name} not found")
        
        history = self.history[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        if 'auc' in history.history:
            axes[1, 0].plot(history.history['auc'], label='Training', linewidth=2)
            axes[1, 0].plot(history.history['val_auc'], label='Validation', linewidth=2)
            axes[1, 0].set_title('Model AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], linewidth=2, color='green')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, model_name: str, 
                             X_test: np.ndarray, 
                             y_test: np.ndarray,
                             save_path: Optional[Path] = None) -> None:
        """
        Plot confusion matrix for a model.
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test targets
            save_path: Path to save the plot
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Get predictions
        y_pred, _ = self.predict(model_name, X_test)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def generate_sample_data(self, n_samples: int = 1000, n_features: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sample data for testing.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
        
        Returns:
            Tuple of (X, y)
        """
        np.random.seed(self.random_state)
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        return X, y

# Main function for testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Neural Network Engine")
    print("=" * 60)
    
    # Initialize engine
    engine = NeuralNetworkEngine()
    print(f"✅ Neural Network Engine initialized")
    print(f"Model directory: {engine.model_dir}")
    
    # Generate sample data
    print("\nGenerating sample data...")
    X, y = engine.generate_sample_data(n_samples=1000, n_features=20)
    print(f"Data shape: {X.shape}, Target shape: {y.shape}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Build and train model
    print("\nBuilding and training model...")
    model = engine.build_mlp(X_train.shape[1], [64, 32], 'relu')
    
    results = engine.train_model(
        X_train, y_train,
        X_val, y_val,
        model_name='test_model',
        model=model,
        epochs=10,  # Small number for testing
        batch_size=32
    )
    
    print(f"\n✅ Model training complete!")
    print(f"Metrics: {results['metrics']}")
    
    # Save model
    print(f"\nSaving model...")
    engine.save_model('test_model')
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
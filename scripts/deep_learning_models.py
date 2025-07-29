import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import joblib

class OptionPricingModels:
    """Deep Learning models for option pricing"""
    
    def __init__(self, random_state: int = 42):
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.models = {}
        self.history = {}
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with option data
        
        Returns:
            Tuple of (features, targets)
        """
        # Select features
        feature_cols = ['S0', 'K', 'T', 'r', 'sigma', 'option_type', 'moneyness', 'time_value']
        X = df[feature_cols].values
        y = df['target_price'].values.reshape(-1, 1)
        
        # Normalize features
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        return X_scaled, y_scaled.ravel()
    
    def create_mlp_model(self, input_dim: int) -> keras.Model:
        """
        Create Multi-Layer Perceptron model
        
        Args:
            input_dim: Number of input features
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_lstm_model(self, sequence_length: int, n_features: int) -> keras.Model:
        """
        Create LSTM model for time series option pricing
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features per timestep
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
            layers.Dropout(0.3),
            
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_transformer_model(self, sequence_length: int, n_features: int) -> keras.Model:
        """
        Create Transformer model for option pricing
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features per timestep
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=(sequence_length, n_features))
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=64
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = layers.Dropout(0.1)(attention_output)
        attention_output = layers.LayerNormalization()(inputs + attention_output)
        
        # Feed Forward Network
        ffn_output = layers.Dense(256, activation='relu')(attention_output)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        ffn_output = layers.Dense(n_features)(ffn_output)
        
        # Add & Norm
        ffn_output = layers.LayerNormalization()(attention_output + ffn_output)
        
        # Global pooling and final layers
        pooled = layers.GlobalAveragePooling1D()(ffn_output)
        outputs = layers.Dense(64, activation='relu')(pooled)
        outputs = layers.Dropout(0.2)(outputs)
        outputs = layers.Dense(1, activation='linear')(outputs)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, model_name: str,
                   epochs: int = 100, batch_size: int = 32) -> keras.Model:
        """
        Train a model with early stopping
        
        Args:
            model: Keras model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_name: Name for saving the model
            epochs: Maximum number of epochs
            batch_size: Batch size for training
        
        Returns:
            Trained model
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.models[model_name] = model
        self.history[model_name] = history
        
        return model
    
    def evaluate_model(self, model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test, y_test: Test data
        
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions and targets
        y_test_orig = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_orig = self.scaler_y.inverse_transform(y_pred).ravel()
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'mae': mean_absolute_error(y_test_orig, y_pred_orig),
            'r2': r2_score(y_test_orig, y_pred_orig),
            'mape': np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
        }
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, model_type: str = 'mlp',
                      k_folds: int = 5) -> Dict[str, List[float]]:
        """
        Perform k-fold cross validation
        
        Args:
            X, y: Data for cross validation
            model_type: Type of model ('mlp', 'lstm', 'transformer')
            k_folds: Number of folds
        
        Returns:
            Dictionary of cross validation scores
        """
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_scores = {'rmse': [], 'mae': [], 'r2': [], 'mape': []}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Training fold {fold + 1}/{k_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create model
            if model_type == 'mlp':
                model = self.create_mlp_model(X.shape[1])
            else:
                raise ValueError(f"Model type {model_type} not supported for cross validation")
            
            # Train model
            model = self.train_model(model, X_train, y_train, X_val, y_val, 
                                   f"{model_type}_fold_{fold}", epochs=50)
            
            # Evaluate
            metrics = self.evaluate_model(model, X_val, y_val)
            for metric, value in metrics.items():
                cv_scores[metric].append(value)
        
        return cv_scores
    
    def plot_training_history(self, model_name: str):
        """Plot training history"""
        if model_name not in self.history:
            print(f"No history found for model {model_name}")
            return
        
        history = self.history[model_name]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # MAE
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, filepath: str):
        """Save trained models and scalers"""
        for name, model in self.models.items():
            model.save(f"{filepath}_{name}.h5")
        
        joblib.dump(self.scaler_X, f"{filepath}_scaler_X.pkl")
        joblib.dump(self.scaler_y, f"{filepath}_scaler_y.pkl")
    
    def load_models(self, filepath: str, model_names: List[str]):
        """Load trained models and scalers"""
        for name in model_names:
            self.models[name] = keras.models.load_model(f"{filepath}_{name}.h5")
        
        self.scaler_X = joblib.load(f"{filepath}_scaler_X.pkl")
        self.scaler_y = joblib.load(f"{filepath}_scaler_y.pkl")

# Training script
if __name__ == "__main__":
    # Load data
    print("Loading training data...")
    df = pd.read_csv('option_training_data.csv')
    
    # Initialize model trainer
    trainer = OptionPricingModels()
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train MLP model
    print("\nTraining MLP model...")
    mlp_model = trainer.create_mlp_model(X.shape[1])
    mlp_model = trainer.train_model(mlp_model, X_train, y_train, X_val, y_val, 'mlp')
    
    # Evaluate MLP
    mlp_metrics = trainer.evaluate_model(mlp_model, X_test, y_test)
    print(f"\nMLP Model Performance:")
    for metric, value in mlp_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Cross validation
    print("\nPerforming cross validation...")
    cv_scores = trainer.cross_validate(X, y, 'mlp', k_folds=5)
    print(f"\nCross Validation Results:")
    for metric, scores in cv_scores.items():
        print(f"{metric.upper()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Save models
    trainer.save_models('trained_models')
    print("\nModels saved successfully!")

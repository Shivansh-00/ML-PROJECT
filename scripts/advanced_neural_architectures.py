import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback
import tensorflow_probability as tfp
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt

class AdvancedNeuralArchitectures:
    """Advanced neural network architectures for option pricing"""
    
    def __init__(self, random_state: int = 42):
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        self.models = {}
        self.histories = {}
    
    def create_attention_mlp(self, input_dim: int, name: str = "attention_mlp") -> Model:
        """
        Create MLP with self-attention mechanism for feature importance
        """
        inputs = layers.Input(shape=(input_dim,), name='input')
        
        # Initial dense layers
        x = layers.Dense(256, activation='relu', name='dense_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(0.3, name='dropout_1')(x)
        
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        # Self-attention mechanism
        # Reshape for attention
        x_reshaped = layers.Reshape((1, 128), name='reshape_for_attention')(x)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=16, name='multi_head_attention'
        )(x_reshaped, x_reshaped)
        
        # Flatten back
        attention_flat = layers.Flatten(name='attention_flatten')(attention_output)
        
        # Combine with original features
        combined = layers.Concatenate(name='combine_attention')([x, attention_flat])
        
        # Final layers
        x = layers.Dense(64, activation='relu', name='dense_3')(combined)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.Dropout(0.2, name='dropout_3')(x)
        
        x = layers.Dense(32, activation='relu', name='dense_4')(x)
        x = layers.Dropout(0.2, name='dropout_4')(x)
        
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_residual_network(self, input_dim: int, name: str = "resnet") -> Model:
        """
        Create ResNet-style architecture for option pricing
        """
        inputs = layers.Input(shape=(input_dim,), name='input')
        
        # Initial layer
        x = layers.Dense(256, activation='relu', name='initial_dense')(inputs)
        x = layers.BatchNormalization(name='initial_bn')(x)
        
        # Residual blocks
        for i in range(3):
            # Store input for residual connection
            residual = x
            
            # First layer of residual block
            x = layers.Dense(256, activation='relu', name=f'res_block_{i}_dense_1')(x)
            x = layers.BatchNormalization(name=f'res_block_{i}_bn_1')(x)
            x = layers.Dropout(0.3, name=f'res_block_{i}_dropout_1')(x)
            
            # Second layer of residual block
            x = layers.Dense(256, activation='relu', name=f'res_block_{i}_dense_2')(x)
            x = layers.BatchNormalization(name=f'res_block_{i}_bn_2')(x)
            
            # Residual connection
            x = layers.Add(name=f'res_block_{i}_add')([x, residual])
            x = layers.Dropout(0.3, name=f'res_block_{i}_dropout_2')(x)
        
        # Final layers
        x = layers.Dense(128, activation='relu', name='final_dense_1')(x)
        x = layers.BatchNormalization(name='final_bn_1')(x)
        x = layers.Dropout(0.2, name='final_dropout_1')(x)
        
        x = layers.Dense(64, activation='relu', name='final_dense_2')(x)
        x = layers.Dropout(0.2, name='final_dropout_2')(x)
        
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_bayesian_network(self, input_dim: int, name: str = "bayesian_nn") -> Model:
        """
        Create Bayesian Neural Network for uncertainty quantification
        """
        # Define prior and posterior functions
        def prior(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            return lambda t: tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(n, dtype=dtype),
                scale_diag=tf.ones(n, dtype=dtype)
            )
        
        def posterior(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=t[..., :n],
                        scale_diag=1e-5 + tf.nn.softplus(0.05 * t[..., n:])
                    )
                )
            ])
        
        inputs = layers.Input(shape=(input_dim,), name='input')
        
        # Bayesian layers
        x = tfp.layers.DenseVariational(
            256, 
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1/1000,  # Adjust based on dataset size
            activation='relu',
            name='bayesian_dense_1'
        )(inputs)
        x = layers.Dropout(0.3, name='dropout_1')(x)
        
        x = tfp.layers.DenseVariational(
            128,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1/1000,
            activation='relu',
            name='bayesian_dense_2'
        )(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        x = tfp.layers.DenseVariational(
            64,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1/1000,
            activation='relu',
            name='bayesian_dense_3'
        )(x)
        x = layers.Dropout(0.2, name='dropout_3')(x)
        
        # Output layer (deterministic)
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        
        # Custom loss function for Bayesian NN
        def bayesian_loss(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=bayesian_loss,
            metrics=['mae']
        )
        
        return model
    
    def create_ensemble_model(self, input_dim: int, n_models: int = 5, 
                            name: str = "ensemble") -> List[Model]:
        """
        Create ensemble of diverse models
        """
        models = []
        
        for i in range(n_models):
            # Create diverse architectures
            if i == 0:
                model = self.create_attention_mlp(input_dim, f"{name}_attention_{i}")
            elif i == 1:
                model = self.create_residual_network(input_dim, f"{name}_resnet_{i}")
            elif i == 2:
                model = self.create_bayesian_network(input_dim, f"{name}_bayesian_{i}")
            else:
                # Standard MLP with different architectures
                model = self._create_diverse_mlp(input_dim, i, f"{name}_mlp_{i}")
            
            models.append(model)
        
        return models
    
    def _create_diverse_mlp(self, input_dim: int, variant: int, name: str) -> Model:
        """Create diverse MLP architectures"""
        inputs = layers.Input(shape=(input_dim,), name='input')
        
        if variant == 3:
            # Wide and shallow
            x = layers.Dense(512, activation='relu', name='dense_1')(inputs)
            x = layers.BatchNormalization(name='bn_1')(x)
            x = layers.Dropout(0.4, name='dropout_1')(x)
            
            x = layers.Dense(256, activation='relu', name='dense_2')(x)
            x = layers.Dropout(0.3, name='dropout_2')(x)
            
        else:  # variant == 4
            # Deep and narrow
            x = layers.Dense(128, activation='relu', name='dense_1')(inputs)
            x = layers.BatchNormalization(name='bn_1')(x)
            x = layers.Dropout(0.2, name='dropout_1')(x)
            
            x = layers.Dense(128, activation='relu', name='dense_2')(x)
            x = layers.BatchNormalization(name='bn_2')(x)
            x = layers.Dropout(0.2, name='dropout_2')(x)
            
            x = layers.Dense(64, activation='relu', name='dense_3')(x)
            x = layers.BatchNormalization(name='bn_3')(x)
            x = layers.Dropout(0.2, name='dropout_3')(x)
            
            x = layers.Dense(64, activation='relu', name='dense_4')(x)
            x = layers.Dropout(0.2, name='dropout_4')(x)
            
            x = layers.Dense(32, activation='relu', name='dense_5')(x)
            x = layers.Dropout(0.2, name='dropout_5')(x)
        
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_advanced_lstm(self, sequence_length: int, n_features: int, 
                           name: str = "advanced_lstm") -> Model:
        """
        Create advanced LSTM with attention and residual connections
        """
        inputs = layers.Input(shape=(sequence_length, n_features), name='input')
        
        # First LSTM layer
        lstm1 = layers.LSTM(128, return_sequences=True, name='lstm_1')(inputs)
        lstm1 = layers.Dropout(0.3, name='lstm_dropout_1')(lstm1)
        
        # Second LSTM layer with residual connection
        lstm2 = layers.LSTM(128, return_sequences=True, name='lstm_2')(lstm1)
        lstm2 = layers.Dropout(0.3, name='lstm_dropout_2')(lstm2)
        
        # Residual connection
        lstm_residual = layers.Add(name='lstm_residual')([lstm1, lstm2])
        
        # Attention mechanism
        attention = layers.MultiHeadAttention(
            num_heads=8, key_dim=16, name='lstm_attention'
        )(lstm_residual, lstm_residual)
        
        # Combine LSTM and attention
        combined = layers.Add(name='combine_lstm_attention')([lstm_residual, attention])
        
        # Global pooling
        avg_pool = layers.GlobalAveragePooling1D(name='global_avg_pool')(combined)
        max_pool = layers.GlobalMaxPooling1D(name='global_max_pool')(combined)
        
        # Combine pooling outputs
        pooled = layers.Concatenate(name='combine_pooling')([avg_pool, max_pool])
        
        # Final dense layers
        x = layers.Dense(64, activation='relu', name='dense_1')(pooled)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(0.2, name='dropout_1')(x)
        
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.2, name='dropout_2')(x)
        
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_transformer_encoder(self, sequence_length: int, n_features: int,
                                 name: str = "transformer_encoder") -> Model:
        """
        Create Transformer encoder for option pricing
        """
        inputs = layers.Input(shape=(sequence_length, n_features), name='input')
        
        # Positional encoding
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        position_embedding = layers.Embedding(
            input_dim=sequence_length, output_dim=n_features, name='position_embedding'
        )(positions)
        
        # Add positional encoding to inputs
        x = layers.Add(name='add_position_encoding')([inputs, position_embedding])
        
        # Multiple transformer blocks
        for i in range(3):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=8, key_dim=64, name=f'attention_{i}'
            )(x, x)
            
            # Add & Norm
            attention_output = layers.Dropout(0.1, name=f'attention_dropout_{i}')(attention_output)
            x1 = layers.Add(name=f'add_attention_{i}')([x, attention_output])
            x1 = layers.LayerNormalization(epsilon=1e-6, name=f'norm_attention_{i}')(x1)
            
            # Feed Forward Network
            ffn_output = layers.Dense(256, activation='relu', name=f'ffn_1_{i}')(x1)
            ffn_output = layers.Dropout(0.1, name=f'ffn_dropout_1_{i}')(ffn_output)
            ffn_output = layers.Dense(n_features, name=f'ffn_2_{i}')(ffn_output)
            
            # Add & Norm
            ffn_output = layers.Dropout(0.1, name=f'ffn_dropout_2_{i}')(ffn_output)
            x = layers.Add(name=f'add_ffn_{i}')([x1, ffn_output])
            x = layers.LayerNormalization(epsilon=1e-6, name=f'norm_ffn_{i}')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D(name='global_pooling')(x)
        
        # Final layers
        x = layers.Dense(128, activation='relu', name='final_dense_1')(x)
        x = layers.Dropout(0.2, name='final_dropout_1')(x)
        
        x = layers.Dense(64, activation='relu', name='final_dense_2')(x)
        x = layers.Dropout(0.2, name='final_dropout_2')(x)
        
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model

class AdvancedTrainingCallbacks:
    """Advanced training callbacks for better model performance"""
    
    @staticmethod
    def get_advanced_callbacks(model_name: str, patience: int = 20) -> List[Callback]:
        """Get comprehensive set of training callbacks"""
        
        callbacks = [
            # Early stopping with restore best weights
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpointing
            keras.callbacks.ModelCheckpoint(
                filepath=f'best_{model_name}.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Learning rate scheduling
            keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * 0.95 ** epoch,
                verbose=0
            ),
            
            # Custom callback for monitoring
            MonitoringCallback()
        ]
        
        return callbacks

class MonitoringCallback(Callback):
    """Custom callback for advanced monitoring"""
    
    def __init__(self):
        super().__init__()
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
        # Check for overfitting
        if epoch > 10:
            recent_train_loss = np.mean(self.losses[-5:])
            recent_val_loss = np.mean(self.val_losses[-5:])
            
            if recent_val_loss > recent_train_loss * 1.5:
                print(f"\nWarning: Potential overfitting detected at epoch {epoch}")
                print(f"Train loss: {recent_train_loss:.6f}, Val loss: {recent_val_loss:.6f}")

# Example usage
if __name__ == "__main__":
    # Initialize architectures
    architectures = AdvancedNeuralArchitectures(random_state=42)
    
    # Create sample data
    input_dim = 50
    sequence_length = 30
    n_features = 10
    
    print("Creating advanced neural architectures...")
    
    # Create attention MLP
    attention_model = architectures.create_attention_mlp(input_dim)
    print(f"Attention MLP created: {attention_model.count_params()} parameters")
    
    # Create ResNet
    resnet_model = architectures.create_residual_network(input_dim)
    print(f"ResNet created: {resnet_model.count_params()} parameters")
    
    # Create Bayesian NN
    try:
        bayesian_model = architectures.create_bayesian_network(input_dim)
        print(f"Bayesian NN created: {bayesian_model.count_params()} parameters")
    except Exception as e:
        print(f"Bayesian NN creation failed: {e}")
    
    # Create ensemble
    ensemble_models = architectures.create_ensemble_model(input_dim, n_models=3)
    print(f"Ensemble created: {len(ensemble_models)} models")
    
    # Create advanced LSTM
    lstm_model = architectures.create_advanced_lstm(sequence_length, n_features)
    print(f"Advanced LSTM created: {lstm_model.count_params()} parameters")
    
    # Create Transformer
    transformer_model = architectures.create_transformer_encoder(sequence_length, n_features)
    print(f"Transformer created: {transformer_model.count_params()} parameters")
    
    print("\nAll architectures created successfully!")

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsembleMethods:
    """Advanced ensemble methods for option pricing with sophisticated aggregation"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.base_models = []
        self.meta_model = None
        self.ensemble_weights = None
        self.model_performances = {}
        self.diversity_metrics = {}
    
    def create_diverse_base_models(self, input_dim: int, n_models: int = 7) -> List[keras.Model]:
        """
        Create diverse base models with different architectures and hyperparameters
        """
        models = []
        
        # Model 1: Deep narrow network
        model1 = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ], name='deep_narrow')
        
        # Model 2: Wide shallow network
        model2 = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='linear')
        ], name='wide_shallow')
        
        # Model 3: Standard architecture with different activation
        model3 = keras.Sequential([
            keras.layers.Dense(256, activation='elu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='elu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='elu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='linear')
        ], name='elu_activation')
        
        # Model 4: Residual-style connections
        inputs = keras.layers.Input(shape=(input_dim,))
        x = keras.layers.Dense(256, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        # Residual block
        residual = x
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Add()([x, residual])
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(1, activation='linear')(x)
        
        model4 = keras.Model(inputs=inputs, outputs=outputs, name='residual_style')
        
        # Model 5: High dropout regularization
        model5 = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(1, activation='linear')
        ], name='high_dropout')
        
        # Model 6: Different optimizer and learning rate
        model6 = keras.Sequential([
            keras.layers.Dense(256, activation='swish', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='swish'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='swish'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='linear')
        ], name='swish_activation')
        
        # Model 7: Ensemble of smaller networks
        model7 = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='linear')
        ], name='compact_network')
        
        models = [model1, model2, model3, model4, model5, model6, model7]
        
        # Compile models with different optimizers and learning rates
        optimizers = [
            keras.optimizers.Adam(learning_rate=0.001),
            keras.optimizers.Adam(learning_rate=0.0005),
            keras.optimizers.RMSprop(learning_rate=0.001),
            keras.optimizers.Adam(learning_rate=0.002),
            keras.optimizers.AdamW(learning_rate=0.001),
            keras.optimizers.Adam(learning_rate=0.0008),
            keras.optimizers.Nadam(learning_rate=0.001)
        ]
        
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
        
        self.base_models = models
        return models
    
    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train all base models with different training strategies
        """
        training_histories = {}
        
        for i, model in enumerate(self.base_models):
            print(f"Training model {i+1}/{len(self.base_models)}: {model.name}")
            
            # Different training strategies for diversity
            if i % 3 == 0:
                # Standard training
                callbacks = [
                    keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8)
                ]
            elif i % 3 == 1:
                # More aggressive early stopping
                callbacks = [
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=5)
                ]
            else:
                # Longer training with cyclic learning rate
                callbacks = [
                    keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                    keras.callbacks.LearningRateScheduler(
                        lambda epoch: 0.001 * (0.95 ** (epoch // 10))
                    )
                ]
            
            # Train with different batch sizes for diversity
            current_batch_size = batch_size if i % 2 == 0 else batch_size * 2
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=current_batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            training_histories[model.name] = history.history
            
            # Evaluate model performance
            val_pred = model.predict(X_val, verbose=0)
            val_mse = mean_squared_error(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            self.model_performances[model.name] = {
                'mse': val_mse,
                'mae': val_mae,
                'r2': val_r2,
                'final_val_loss': history.history['val_loss'][-1]
            }
            
            print(f"  Validation MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}")
        
        return training_histories
    
    def calculate_ensemble_weights(self, X_val: np.ndarray, y_val: np.ndarray,
                                 method: str = 'performance_based') -> np.ndarray:
        """
        Calculate optimal ensemble weights using different strategies
        """
        n_models = len(self.base_models)
        
        if method == 'equal':
            weights = np.ones(n_models) / n_models
            
        elif method == 'performance_based':
            # Weight based on inverse validation error
            val_errors = []
            for model in self.base_models:
                pred = model.predict(X_val, verbose=0)
                error = mean_squared_error(y_val, pred)
                val_errors.append(error)
            
            # Convert to weights (inverse of error)
            val_errors = np.array(val_errors)
            weights = 1 / (val_errors + 1e-8)
            weights = weights / np.sum(weights)
            
        elif method == 'diversity_weighted':
            # Weight based on diversity and performance
            predictions = np.array([model.predict(X_val, verbose=0).flatten() 
                                  for model in self.base_models])
            
            # Calculate pairwise correlations
            correlations = np.corrcoef(predictions)
            
            # Calculate diversity score (1 - average correlation with others)
            diversity_scores = []
            for i in range(n_models):
                avg_corr = np.mean([correlations[i, j] for j in range(n_models) if i != j])
                diversity_scores.append(1 - avg_corr)
            
            # Combine with performance
            performance_scores = [1 / (self.model_performances[model.name]['mse'] + 1e-8) 
                                for model in self.base_models]
            
            # Weighted combination
            combined_scores = np.array(diversity_scores) * np.array(performance_scores)
            weights = combined_scores / np.sum(combined_scores)
            
        elif method == 'optimal':
            # Solve optimization problem for optimal weights
            from scipy.optimize import minimize
            
            def objective(w):
                w = w / np.sum(w)  # Normalize weights
                ensemble_pred = np.zeros(len(y_val))
                for i, model in enumerate(self.base_models):
                    pred = model.predict(X_val, verbose=0).flatten()
                    ensemble_pred += w[i] * pred
                return mean_squared_error(y_val, ensemble_pred)
            
            # Constraints: weights sum to 1 and are non-negative
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(n_models)]
            
            # Initial guess: equal weights
            x0 = np.ones(n_models) / n_models
            
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            weights = result.x
            
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        self.ensemble_weights = weights
        return weights
    
    def create_stacking_ensemble(self, input_dim: int, n_base_models: int) -> keras.Model:
        """
        Create a meta-learner for stacking ensemble
        """
        # Meta-learner takes predictions from base models as input
        meta_model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(n_base_models,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='linear')
        ], name='meta_learner')
        
        meta_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.meta_model = meta_model
        return meta_model
    
    def train_stacking_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train stacking ensemble using cross-validation
        """
        # Generate out-of-fold predictions for training meta-learner
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Store out-of-fold predictions
        oof_predictions = np.zeros((len(X_train), len(self.base_models)))
        val_predictions = np.zeros((len(X_val), len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"Training fold {fold + 1}/{cv_folds}")
            
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            fold_val_preds = []
            
            for i, model in enumerate(self.base_models):
                # Clone and train model on fold
                model_clone = keras.models.clone_model(model)
                model_clone.compile(
                    optimizer=model.optimizer,
                    loss=model.loss,
                    metrics=model.metrics
                )
                
                # Train on fold
                model_clone.fit(
                    X_fold_train, y_fold_train,
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    callbacks=[
                        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                    ]
                )
                
                # Predict on validation fold
                fold_pred = model_clone.predict(X_fold_val, verbose=0).flatten()
                oof_predictions[val_idx, i] = fold_pred
                
                # Predict on actual validation set
                val_pred = model_clone.predict(X_val, verbose=0).flatten()
                fold_val_preds.append(val_pred)
            
            # Average validation predictions across folds
            for i, val_pred in enumerate(fold_val_preds):
                val_predictions[:, i] += val_pred / cv_folds
        
        # Train meta-learner on out-of-fold predictions
        print("Training meta-learner...")
        meta_history = self.meta_model.fit(
            oof_predictions, y_train,
            validation_data=(val_predictions, y_val),
            epochs=100,
            batch_size=64,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8)
            ],
            verbose=1
        )
        
        return {
            'meta_history': meta_history.history,
            'oof_predictions': oof_predictions,
            'val_predictions': val_predictions
        }
    
    def predict_ensemble(self, X: np.ndarray, method: str = 'weighted_average') -> np.ndarray:
        """
        Make ensemble predictions using specified method
        """
        if method == 'weighted_average':
            if self.ensemble_weights is None:
                raise ValueError("Ensemble weights not calculated. Call calculate_ensemble_weights first.")
            
            ensemble_pred = np.zeros(len(X))
            for i, model in enumerate(self.base_models):
                pred = model.predict(X, verbose=0).flatten()
                ensemble_pred += self.ensemble_weights[i] * pred
            
            return ensemble_pred
            
        elif method == 'stacking':
            if self.meta_model is None:
                raise ValueError("Meta-model not trained. Call train_stacking_ensemble first.")
            
            # Get base model predictions
            base_predictions = np.array([model.predict(X, verbose=0).flatten() 
                                       for model in self.base_models]).T
            
            # Meta-learner prediction
            ensemble_pred = self.meta_model.predict(base_predictions, verbose=0).flatten()
            return ensemble_pred
            
        elif method == 'median':
            predictions = np.array([model.predict(X, verbose=0).flatten() 
                                  for model in self.base_models])
            return np.median(predictions, axis=0)
            
        elif method == 'trimmed_mean':
            predictions = np.array([model.predict(X, verbose=0).flatten() 
                                  for model in self.base_models])
            # Remove top and bottom 20% and average
            return stats.trim_mean(predictions, 0.2, axis=0)
            
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def calculate_diversity_metrics(self, X_val: np.ndarray) -> Dict[str, float]:
        """
        Calculate ensemble diversity metrics
        """
        # Get predictions from all models
        predictions = np.array([model.predict(X_val, verbose=0).flatten() 
                              for model in self.base_models])
        
        # Pairwise correlations
        correlations = np.corrcoef(predictions)
        avg_correlation = np.mean(correlations[np.triu_indices_from(correlations, k=1)])
        
        # Disagreement measure
        mean_pred = np.mean(predictions, axis=0)
        disagreement = np.mean([np.mean((pred - mean_pred)**2) for pred in predictions])
        
        # Diversity index (1 - average correlation)
        diversity_index = 1 - avg_correlation
        
        # Q-statistic (measure of diversity between pairs)
        q_statistics = []
        n_models = len(self.base_models)
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                pred_i, pred_j = predictions[i], predictions[j]
                
                # Convert to binary classification for Q-statistic
                median_i, median_j = np.median(pred_i), np.median(pred_j)
                binary_i = (pred_i > median_i).astype(int)
                binary_j = (pred_j > median_j).astype(int)
                
                # Calculate Q-statistic
                n11 = np.sum((binary_i == 1) & (binary_j == 1))
                n10 = np.sum((binary_i == 1) & (binary_j == 0))
                n01 = np.sum((binary_i == 0) & (binary_j == 1))
                n00 = np.sum((binary_i == 0) & (binary_j == 0))
                
                if (n11 * n00 + n01 * n10) > 0:
                    q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
                    q_statistics.append(q)
        
        avg_q_statistic = np.mean(q_statistics) if q_statistics else 0
        
        self.diversity_metrics = {
            'average_correlation': avg_correlation,
            'disagreement': disagreement,
            'diversity_index': diversity_index,
            'q_statistic': avg_q_statistic
        }
        
        return self.diversity_metrics
    
    def plot_ensemble_analysis(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Create comprehensive ensemble analysis plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ensemble Analysis', fontsize=16)
        
        # 1. Model performance comparison
        model_names = [model.name for model in self.base_models]
        mse_scores = [self.model_performances[name]['mse'] for name in model_names]
        
        axes[0, 0].bar(range(len(model_names)), mse_scores)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Validation MSE')
        axes[0, 0].set_title('Individual Model Performance')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        
        # 2. Ensemble weights
        if self.ensemble_weights is not None:
            axes[0, 1].bar(range(len(model_names)), self.ensemble_weights)
            axes[0, 1].set_xlabel('Models')
            axes[0, 1].set_ylabel('Ensemble Weight')
            axes[0, 1].set_title('Ensemble Weights')
            axes[0, 1].set_xticks(range(len(model_names)))
            axes[0, 1].set_xticklabels(model_names, rotation=45)
        
        # 3. Prediction correlations heatmap
        predictions = np.array([model.predict(X_val, verbose=0).flatten() 
                              for model in self.base_models])
        correlations = np.corrcoef(predictions)
        
        im = axes[0, 2].imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 2].set_title('Model Prediction Correlations')
        axes[0, 2].set_xticks(range(len(model_names)))
        axes[0, 2].set_yticks(range(len(model_names)))
        axes[0, 2].set_xticklabels(model_names, rotation=45)
        axes[0, 2].set_yticklabels(model_names)
        plt.colorbar(im, ax=axes[0, 2])
        
        # 4. Ensemble vs individual predictions
        if self.ensemble_weights is not None:
            ensemble_pred = self.predict_ensemble(X_val, method='weighted_average')
            
            axes[1, 0].scatter(y_val, ensemble_pred, alpha=0.6, label='Ensemble')
            
            # Show best individual model
            best_model_idx = np.argmin(mse_scores)
            best_pred = self.base_models[best_model_idx].predict(X_val, verbose=0).flatten()
            axes[1, 0].scatter(y_val, best_pred, alpha=0.6, label=f'Best Individual ({model_names[best_model_idx]})')
            
            axes[1, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
            axes[1, 0].set_xlabel('True Values')
            axes[1, 0].set_ylabel('Predictions')
            axes[1, 0].set_title('Ensemble vs Best Individual Model')
            axes[1, 0].legend()
        
        # 5. Diversity metrics
        diversity_metrics = self.calculate_diversity_metrics(X_val)
        metric_names = list(diversity_metrics.keys())
        metric_values = list(diversity_metrics.values())
        
        axes[1, 1].bar(range(len(metric_names)), metric_values)
        axes[1, 1].set_xlabel('Diversity Metrics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Ensemble Diversity Metrics')
        axes[1, 1].set_xticks(range(len(metric_names)))
        axes[1, 1].set_xticklabels(metric_names, rotation=45)
        
        # 6. Prediction distribution comparison
        axes[1, 2].hist([pred.flatten() for pred in predictions], 
                       bins=30, alpha=0.7, label=model_names)
        axes[1, 2].set_xlabel('Prediction Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Prediction Distributions')
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_ensemble_methods(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate different ensemble methods
        """
        results = {}
        
        # Individual model performance
        for i, model in enumerate(self.base_models):
            pred = model.predict(X_test, verbose=0).flatten()
            results[f'Model_{i+1}_{model.name}'] = {
                'mse': mean_squared_error(y_test, pred),
                'mae': mean_absolute_error(y_test, pred),
                'r2': r2_score(y_test, pred)
            }
        
        # Ensemble methods
        ensemble_methods = ['weighted_average', 'median', 'trimmed_mean']
        
        if self.meta_model is not None:
            ensemble_methods.append('stacking')
        
        for method in ensemble_methods:
            try:
                pred = self.predict_ensemble(X_test, method=method)
                results[f'Ensemble_{method}'] = {
                    'mse': mean_squared_error(y_test, pred),
                    'mae': mean_absolute_error(y_test, pred),
                    'r2': r2_score(y_test, pred)
                }
            except Exception as e:
                print(f"Error with {method}: {e}")
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize ensemble methods
    ensemble = AdvancedEnsembleMethods(random_state=42)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 5000
    input_dim = 30
    
    X = np.random.randn(n_samples, input_dim)
    y = np.sum(X[:, :5], axis=1) + 0.1 * np.random.randn(n_samples)  # Simple linear relationship with noise
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create and train base models
    print("\nCreating diverse base models...")
    base_models = ensemble.create_diverse_base_models(input_dim, n_models=5)  # Reduced for demo
    
    print("\nTraining base models...")
    training_histories = ensemble.train_base_models(X_train, y_train, X_val, y_val, epochs=50)
    
    # Calculate ensemble weights
    print("\nCalculating ensemble weights...")
    weights = ensemble.calculate_ensemble_weights(X_val, y_val, method='diversity_weighted')
    print(f"Ensemble weights: {weights}")
    
    # Train stacking ensemble
    print("\nTraining stacking ensemble...")
    meta_model = ensemble.create_stacking_ensemble(input_dim, len(base_models))
    stacking_results = ensemble.train_stacking_ensemble(X_train, y_train, X_val, y_val, cv_folds=3)
    
    # Evaluate all methods
    print("\nEvaluating ensemble methods...")
    evaluation_results = ensemble.evaluate_ensemble_methods(X_test, y_test)
    
    print("\nEvaluation Results:")
    for method, metrics in evaluation_results.items():
        print(f"{method}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        print()
    
    # Plot analysis
    ensemble.plot_ensemble_analysis(X_val, y_val)
    
    print("Advanced ensemble methods demonstration complete!")

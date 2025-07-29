import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

class UncertaintyQuantification:
    """Advanced uncertainty quantification for option pricing models"""
    
    def __init__(self):
        self.calibration_data = {}
        self.uncertainty_metrics = {}
    
    def monte_carlo_dropout(self, model: keras.Model, X: np.ndarray, 
                          n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Monte Carlo Dropout for uncertainty estimation
        
        Args:
            model: Trained model with dropout layers
            X: Input data
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        # Enable dropout during inference
        predictions = []
        
        for _ in range(n_samples):
            # Set training=True to enable dropout
            pred = model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred.flatten(), uncertainty.flatten()
    
    def deep_ensemble_uncertainty(self, models: List[keras.Model], 
                                X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate uncertainty using deep ensemble approach
        
        Args:
            models: List of trained models
            X: Input data
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        predictions = []
        
        for model in models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions)
        
        # Calculate ensemble mean and uncertainty
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
    
    def bayesian_uncertainty(self, model: keras.Model, X: np.ndarray,
                           n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract uncertainty from Bayesian neural network
        
        Args:
            model: Bayesian neural network
            X: Input data
            n_samples: Number of samples from posterior
            
        Returns:
            Tuple of (mean_predictions, aleatoric_uncertainty, epistemic_uncertainty)
        """
        predictions = []
        
        for _ in range(n_samples):
            pred = model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calculate different types of uncertainty
        mean_pred = np.mean(predictions, axis=0).flatten()
        total_uncertainty = np.std(predictions, axis=0).flatten()
        
        # For simplicity, assume epistemic uncertainty is the total uncertainty
        # In practice, you would separate aleatoric and epistemic components
        epistemic_uncertainty = total_uncertainty
        aleatoric_uncertainty = np.zeros_like(epistemic_uncertainty)
        
        return mean_pred, aleatoric_uncertainty, epistemic_uncertainty
    
    def quantile_regression_uncertainty(self, X_train: np.ndarray, y_train: np.ndarray,
                                      X_test: np.ndarray, input_dim: int,
                                      quantiles: List[float] = [0.05, 0.5, 0.95]) -> Dict[str, np.ndarray]:
        """
        Use quantile regression for uncertainty estimation
        
        Args:
            X_train, y_train: Training data
            X_test: Test data
            input_dim: Input dimension
            quantiles: Quantiles to predict
            
        Returns:
            Dictionary with quantile predictions
        """
        def quantile_loss(q):
            def loss(y_true, y_pred):
                error = y_true - y_pred
                return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
            return loss
        
        quantile_models = {}
        predictions = {}
        
        for q in quantiles:
            # Create model for each quantile
            model = keras.Sequential([
                keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer='adam',
                loss=quantile_loss(q),
                metrics=['mae']
            )
            
            # Train model
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Make predictions
            pred = model.predict(X_test, verbose=0)
            predictions[f'quantile_{q}'] = pred.flatten()
            quantile_models[q] = model
        
        return predictions
    
    def calculate_prediction_intervals(self, mean_pred: np.ndarray, 
                                     uncertainty: np.ndarray,
                                     confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals from mean and uncertainty
        
        Args:
            mean_pred: Mean predictions
            uncertainty: Uncertainty estimates
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        # Calculate z-score for confidence level
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bounds = mean_pred - z_score * uncertainty
        upper_bounds = mean_pred + z_score * uncertainty
        
        return lower_bounds, upper_bounds
    
    def evaluate_uncertainty_calibration(self, y_true: np.ndarray, 
                                       mean_pred: np.ndarray,
                                       uncertainty: np.ndarray,
                                       n_bins: int = 10) -> Dict[str, float]:
        """
        Evaluate uncertainty calibration using reliability diagrams
        
        Args:
            y_true: True values
            mean_pred: Mean predictions
            uncertainty: Uncertainty estimates
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary with calibration metrics
        """
        # Calculate prediction intervals
        lower_95, upper_95 = self.calculate_prediction_intervals(mean_pred, uncertainty, 0.95)
        lower_90, upper_90 = self.calculate_prediction_intervals(mean_pred, uncertainty, 0.90)
        lower_68, upper_68 = self.calculate_prediction_intervals(mean_pred, uncertainty, 0.68)
        
        # Calculate coverage
        coverage_95 = np.mean((y_true >= lower_95) & (y_true <= upper_95))
        coverage_90 = np.mean((y_true >= lower_90) & (y_true <= upper_90))
        coverage_68 = np.mean((y_true >= lower_68) & (y_true <= upper_68))
        
        # Calculate interval width
        width_95 = np.mean(upper_95 - lower_95)
        width_90 = np.mean(upper_90 - lower_90)
        width_68 = np.mean(upper_68 - lower_68)
        
        # Calculate calibration error
        calibration_error = self._calculate_calibration_error(
            y_true, mean_pred, uncertainty, n_bins
        )
        
        # Calculate sharpness (average uncertainty)
        sharpness = np.mean(uncertainty)
        
        metrics = {
            'coverage_95': coverage_95,
            'coverage_90': coverage_90,
            'coverage_68': coverage_68,
            'width_95': width_95,
            'width_90': width_90,
            'width_68': width_68,
            'calibration_error': calibration_error,
            'sharpness': sharpness,
            'rmse': np.sqrt(mean_squared_error(y_true, mean_pred)),
            'mae': mean_absolute_error(y_true, mean_pred)
        }
        
        return metrics
    
    def _calculate_calibration_error(self, y_true: np.ndarray, mean_pred: np.ndarray,
                                   uncertainty: np.ndarray, n_bins: int) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        # Normalize errors by uncertainty
        normalized_errors = np.abs(y_true - mean_pred) / (uncertainty + 1e-8)
        
        # Create bins based on uncertainty
        bin_boundaries = np.linspace(0, np.max(uncertainty), n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (uncertainty > bin_lower) & (uncertainty <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence for this bin
                accuracy_in_bin = (normalized_errors[in_bin] <= 1).mean()
                avg_confidence_in_bin = (1 / (1 + uncertainty[in_bin])).mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def plot_uncertainty_analysis(self, y_true: np.ndarray, mean_pred: np.ndarray,
                                uncertainty: np.ndarray, title: str = "Uncertainty Analysis"):
        """
        Create comprehensive uncertainty analysis plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16)
        
        # 1. Predictions vs True with uncertainty
        axes[0, 0].scatter(y_true, mean_pred, c=uncertainty, cmap='viridis', alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predictions')
        axes[0, 0].set_title('Predictions vs True (colored by uncertainty)')
        cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
        cbar.set_label('Uncertainty')
        
        # 2. Uncertainty vs Absolute Error
        abs_error = np.abs(y_true - mean_pred)
        axes[0, 1].scatter(uncertainty, abs_error, alpha=0.6)
        axes[0, 1].set_xlabel('Uncertainty')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].set_title('Uncertainty vs Absolute Error')
        
        # Add correlation coefficient
        corr = np.corrcoef(uncertainty, abs_error)[0, 1]
        axes[0, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[0, 1].transAxes, verticalalignment='top')
        
        # 3. Uncertainty distribution
        axes[0, 2].hist(uncertainty, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Uncertainty')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Uncertainty Distribution')
        
        # 4. Prediction intervals
        sorted_indices = np.argsort(mean_pred)
        sorted_true = y_true[sorted_indices]
        sorted_pred = mean_pred[sorted_indices]
        sorted_uncertainty = uncertainty[sorted_indices]
        
        lower_95, upper_95 = self.calculate_prediction_intervals(sorted_pred, sorted_uncertainty, 0.95)
        
        axes[1, 0].fill_between(range(len(sorted_pred)), lower_95, upper_95, 
                               alpha=0.3, label='95% PI')
        axes[1, 0].plot(sorted_true, 'ro', markersize=2, alpha=0.6, label='True')
        axes[1, 0].plot(sorted_pred, 'b-', linewidth=1, label='Predicted')
        axes[1, 0].set_xlabel('Sample Index (sorted by prediction)')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Prediction Intervals')
        axes[1, 0].legend()
        
        # 5. Calibration plot
        self._plot_calibration(y_true, mean_pred, uncertainty, axes[1, 1])
        
        # 6. Residuals vs Uncertainty
        residuals = y_true - mean_pred
        axes[1, 2].scatter(uncertainty, residuals, alpha=0.6)
        axes[1, 2].axhline(y=0, color='r', linestyle='--')
        axes[1, 2].set_xlabel('Uncertainty')
        axes[1, 2].set_ylabel('Residuals')
        axes[1, 2].set_title('Residuals vs Uncertainty')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_calibration(self, y_true: np.ndarray, mean_pred: np.ndarray,
                         uncertainty: np.ndarray, ax):
        """Plot calibration curve"""
        # Calculate different confidence levels
        confidence_levels = np.linspace(0.1, 0.9, 9)
        observed_frequencies = []
        
        for conf_level in confidence_levels:
            lower, upper = self.calculate_prediction_intervals(mean_pred, uncertainty, conf_level)
            coverage = np.mean((y_true >= lower) & (y_true <= upper))
            observed_frequencies.append(coverage)
        
        ax.plot(confidence_levels, observed_frequencies, 'bo-', label='Observed')
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax.set_xlabel('Expected Frequency')
        ax.set_ylabel('Observed Frequency')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def risk_aware_evaluation(self, y_true: np.ndarray, mean_pred: np.ndarray,
                            uncertainty: np.ndarray, risk_aversion: float = 1.0) -> Dict[str, float]:
        """
        Evaluate model performance with risk-aware metrics
        
        Args:
            y_true: True values
            mean_pred: Mean predictions
            uncertainty: Uncertainty estimates
            risk_aversion: Risk aversion parameter (higher = more risk averse)
            
        Returns:
            Dictionary with risk-aware metrics
        """
        # Standard metrics
        mse = mean_squared_error(y_true, mean_pred)
        mae = mean_absolute_error(y_true, mean_pred)
        
        # Risk-adjusted metrics
        # Penalize high uncertainty predictions more
        risk_adjusted_mse = mse + risk_aversion * np.mean(uncertainty**2)
        risk_adjusted_mae = mae + risk_aversion * np.mean(uncertainty)
        
        # Value at Risk (VaR) style metrics
        errors = np.abs(y_true - mean_pred)
        var_95 = np.percentile(errors, 95)
        var_99 = np.percentile(errors, 99)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = np.mean(errors[errors >= var_95])
        es_99 = np.mean(errors[errors >= var_99])
        
        # Uncertainty-weighted metrics
        weights = 1 / (1 + uncertainty)  # Higher weight for low uncertainty predictions
        weighted_mse = np.average((y_true - mean_pred)**2, weights=weights)
        weighted_mae = np.average(np.abs(y_true - mean_pred), weights=weights)
        
        return {
            'mse': mse,
            'mae': mae,
            'risk_adjusted_mse': risk_adjusted_mse,
            'risk_adjusted_mae': risk_adjusted_mae,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'weighted_mse': weighted_mse,
            'weighted_mae': weighted_mae,
            'mean_uncertainty': np.mean(uncertainty),
            'max_uncertainty': np.max(uncertainty)
        }

# Example usage
if __name__ == "__main__":
    # Initialize uncertainty quantification
    uq = UncertaintyQuantification()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate predictions with uncertainty
    y_true = np.random.normal(10, 2, n_samples)
    mean_pred = y_true + np.random.normal(0, 0.5, n_samples)  # Add some prediction error
    uncertainty = np.abs(np.random.normal(1, 0.3, n_samples))  # Simulate uncertainty
    
    print("Evaluating uncertainty quantification...")
    
    # Calculate prediction intervals
    lower_95, upper_95 = uq.calculate_prediction_intervals(mean_pred, uncertainty, 0.95)
    print(f"95% Prediction Interval Width: {np.mean(upper_95 - lower_95):.4f}")
    
    # Evaluate calibration
    calibration_metrics = uq.evaluate_uncertainty_calibration(y_true, mean_pred, uncertainty)
    print("\nCalibration Metrics:")
    for metric, value in calibration_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Risk-aware evaluation
    risk_metrics = uq.risk_aware_evaluation(y_true, mean_pred, uncertainty, risk_aversion=1.5)
    print("\nRisk-Aware Metrics:")
    for metric, value in risk_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot uncertainty analysis
    uq.plot_uncertainty_analysis(y_true, mean_pred, uncertainty, "Sample Uncertainty Analysis")
    
    print("\nUncertainty quantification analysis complete!")

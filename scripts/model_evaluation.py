import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self):
        self.results = {}
    
    def compare_predictions(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                          model_names: List[str]) -> pd.DataFrame:
        """
        Compare predictions from different models
        
        Args:
            y_true: True option prices
            predictions: Dictionary of model predictions
            model_names: List of model names
        
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for name in model_names:
            if name in predictions:
                y_pred = predictions[name]
                
                metrics = {
                    'Model': name,
                    'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'MAE': mean_absolute_error(y_true, y_pred),
                    'R²': r2_score(y_true, y_pred),
                    'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
                    'Max Error': np.max(np.abs(y_true - y_pred)),
                    'Mean Bias': np.mean(y_pred - y_true)
                }
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_prediction_comparison(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                                 model_names: List[str], sample_size: int = 1000):
        """
        Plot prediction vs actual comparison
        """
        n_models = len(model_names)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
        if n_models == 1:
            axes = [axes]
        axes = axes.flatten()
        
        # Sample data for visualization
        indices = np.random.choice(len(y_true), min(sample_size, len(y_true)), replace=False)
        y_sample = y_true[indices]
        
        for i, name in enumerate(model_names):
            if name in predictions:
                y_pred_sample = predictions[name][indices]
                
                axes[i].scatter(y_sample, y_pred_sample, alpha=0.6, s=20)
                axes[i].plot([y_sample.min(), y_sample.max()], 
                           [y_sample.min(), y_sample.max()], 'r--', lw=2)
                
                r2 = r2_score(y_sample, y_pred_sample)
                axes[i].set_title(f'{name} (R² = {r2:.3f})')
                axes[i].set_xlabel('Actual Price')
                axes[i].set_ylabel('Predicted Price')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(model_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                      model_names: List[str]):
        """
        Plot residual analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for name in model_names:
            if name in predictions:
                residuals = y_true - predictions[name]
                
                # Residuals vs Predicted
                axes[0, 0].scatter(predictions[name], residuals, alpha=0.6, label=name)
                axes[0, 0].axhline(y=0, color='r', linestyle='--')
                axes[0, 0].set_xlabel('Predicted Values')
                axes[0, 0].set_ylabel('Residuals')
                axes[0, 0].set_title('Residuals vs Predicted')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Histogram of residuals
                axes[0, 1].hist(residuals, bins=50, alpha=0.7, label=name)
                axes[0, 1].set_xlabel('Residuals')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Distribution of Residuals')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot for first model
        if model_names and model_names[0] in predictions:
            from scipy import stats
            residuals = y_true - predictions[model_names[0]]
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title(f'Q-Q Plot - {model_names[0]}')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Error by option moneyness
        if len(model_names) > 0 and model_names[0] in predictions:
            # Assuming we have access to moneyness data
            axes[1, 1].set_title('Error Analysis by Moneyness')
            axes[1, 1].set_xlabel('Moneyness (S/K)')
            axes[1, 1].set_ylabel('Absolute Error')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def sensitivity_analysis(self, model, scaler_X, scaler_y, base_params: Dict[str, float]):
        """
        Perform sensitivity analysis (Greeks approximation)
        
        Args:
            model: Trained model
            scaler_X, scaler_y: Fitted scalers
            base_params: Base option parameters
        
        Returns:
            Dictionary with sensitivity measures
        """
        sensitivities = {}
        
        # Parameter ranges for sensitivity
        param_ranges = {
            'S0': np.linspace(base_params['S0'] * 0.8, base_params['S0'] * 1.2, 50),
            'sigma': np.linspace(base_params['sigma'] * 0.5, base_params['sigma'] * 1.5, 50),
            'T': np.linspace(0.01, base_params['T'] * 2, 50),
            'r': np.linspace(0.01, base_params['r'] * 2, 50)
        }
        
        for param, values in param_ranges.items():
            prices = []
            
            for value in values:
                # Create parameter vector
                params = base_params.copy()
                params[param] = value
                
                # Convert to feature vector
                feature_vector = np.array([[
                    params['S0'], params['K'], params['T'], params['r'], 
                    params['sigma'], params['option_type'], 
                    params['S0']/params['K'], params['T']
                ]])
                
                # Scale and predict
                feature_scaled = scaler_X.transform(feature_vector)
                price_scaled = model.predict(feature_scaled)
                price = scaler_y.inverse_transform(price_scaled.reshape(-1, 1))[0, 0]
                prices.append(price)
            
            sensitivities[param] = {
                'values': values,
                'prices': np.array(prices)
            }
        
        return sensitivities
    
    def plot_sensitivity_analysis(self, sensitivities: Dict[str, Dict[str, np.ndarray]]):
        """
        Plot sensitivity analysis results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        param_labels = {
            'S0': 'Stock Price ($)',
            'sigma': 'Volatility',
            'T': 'Time to Maturity (years)',
            'r': 'Risk-free Rate'
        }
        
        for i, (param, data) in enumerate(sensitivities.items()):
            if i < 4:
                axes[i].plot(data['values'], data['prices'], 'b-', linewidth=2)
                axes[i].set_xlabel(param_labels.get(param, param))
                axes[i].set_ylabel('Option Price ($)')
                axes[i].set_title(f'Sensitivity to {param_labels.get(param, param)}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_comparison(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                                    model_names: List[str]) -> go.Figure:
        """
        Create interactive Plotly comparison chart
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction vs Actual', 'Residuals Distribution', 
                          'Error by Price Range', 'Model Performance Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, name in enumerate(model_names):
            if name in predictions:
                color = colors[i % len(colors)]
                y_pred = predictions[name]
                
                # Prediction vs Actual
                fig.add_trace(
                    go.Scatter(
                        x=y_true[:1000], y=y_pred[:1000],
                        mode='markers',
                        name=f'{name}',
                        marker=dict(color=color, opacity=0.6),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Residuals
                residuals = y_true - y_pred
                fig.add_trace(
                    go.Histogram(
                        x=residuals,
                        name=f'{name} Residuals',
                        marker=dict(color=color, opacity=0.7),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # Add perfect prediction line
        min_val, max_val = y_true.min(), y_true.max()
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Performance metrics table
        metrics_df = self.compare_predictions(y_true, predictions, model_names)
        fig.add_trace(
            go.Table(
                header=dict(values=list(metrics_df.columns)),
                cells=dict(values=[metrics_df[col] for col in metrics_df.columns])
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Model Comparison Dashboard")
        return fig
    
    def generate_report(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                       model_names: List[str]) -> str:
        """
        Generate comprehensive evaluation report
        """
        report = "# Option Pricing Model Evaluation Report\n\n"
        
        # Model comparison
        comparison_df = self.compare_predictions(y_true, predictions, model_names)
        report += "## Model Performance Comparison\n\n"
        report += comparison_df.to_markdown(index=False)
        report += "\n\n"
        
        # Best model analysis
        best_model = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
        report += f"## Best Performing Model: {best_model}\n\n"
        
        best_metrics = comparison_df[comparison_df['Model'] == best_model].iloc[0]
        report += f"- **R² Score**: {best_metrics['R²']:.4f}\n"
        report += f"- **RMSE**: ${best_metrics['RMSE']:.4f}\n"
        report += f"- **MAE**: ${best_metrics['MAE']:.4f}\n"
        report += f"- **MAPE**: {best_metrics['MAPE']:.2f}%\n\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        if best_metrics['R²'] > 0.95:
            report += "- Excellent model performance with high accuracy\n"
        elif best_metrics['R²'] > 0.90:
            report += "- Good model performance, suitable for production use\n"
        else:
            report += "- Model performance needs improvement\n"
        
        if best_metrics['MAPE'] < 5:
            report += "- Low prediction error, reliable for pricing\n"
        else:
            report += "- Consider additional feature engineering or model tuning\n"
        
        return report

# Example usage
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    
    # This would be called with actual model predictions
    print("Model evaluator initialized. Use with trained models for comprehensive evaluation.")

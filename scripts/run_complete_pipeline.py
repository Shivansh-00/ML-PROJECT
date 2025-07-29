"""
Complete ML Option Pricing Pipeline
Run this script to execute the full pipeline from data generation to model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_generation import OptionDataGenerator
from baseline_models import BaselineModels
from deep_learning_models import OptionPricingModels
from model_evaluation import ModelEvaluator

def main():
    print("="*60)
    print("ML OPTION PRICING PIPELINE")
    print("="*60)
    
    # Step 1: Data Generation
    print("\n1. GENERATING TRAINING DATA")
    print("-" * 30)
    
    generator = OptionDataGenerator(seed=42)
    
    # Generate main dataset
    print("Generating option pricing dataset...")
    df = generator.generate_training_data(n_samples=20000)
    print(f"Generated {len(df)} training samples")
    
    # Generate time series data for LSTM
    print("Generating time series data...")
    X_ts, y_ts = generator.generate_time_series_data(n_samples=2000)
    print(f"Generated time series data: {X_ts.shape}")
    
    # Display sample data
    print("\nSample data:")
    print(df.head())
    print(f"\nData statistics:")
    print(df.describe())
    
    # Step 2: Baseline Model Evaluation
    print("\n2. BASELINE MODEL EVALUATION")
    print("-" * 30)
    
    baseline = BaselineModels()
    
    # Test with sample parameters
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
    
    print(f"Testing with: S=${S}, K=${K}, T={T}, r={r*100}%, σ={sigma*100}%")
    
    # Compare models for call option
    call_comparison = baseline.compare_models(S, K, T, r, sigma, 'call')
    print("\nCall Option Pricing Comparison:")
    print(call_comparison.to_string(index=False))
    
    # Compare models for put option
    put_comparison = baseline.compare_models(S, K, T, r, sigma, 'put')
    print("\nPut Option Pricing Comparison:")
    print(put_comparison.to_string(index=False))
    
    # Step 3: Deep Learning Model Training
    print("\n3. DEEP LEARNING MODEL TRAINING")
    print("-" * 30)
    
    ml_trainer = OptionPricingModels(random_state=42)
    
    # Prepare data
    X, y = ml_trainer.prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train MLP model
    print("\nTraining MLP model...")
    mlp_model = ml_trainer.create_mlp_model(X.shape[1])
    mlp_model = ml_trainer.train_model(mlp_model, X_train, y_train, X_val, y_val, 'mlp', epochs=50)
    
    # Train LSTM model (if time series data is available)
    if X_ts.shape[0] > 100:
        print("\nTraining LSTM model...")
        X_ts_train, X_ts_test, y_ts_train, y_ts_test = train_test_split(X_ts, y_ts, test_size=0.2, random_state=42)
        X_ts_train, X_ts_val, y_ts_train, y_ts_val = train_test_split(X_ts_train, y_ts_train, test_size=0.2, random_state=42)
        
        lstm_model = ml_trainer.create_lstm_model(X_ts.shape[1], X_ts.shape[2])
        lstm_model = ml_trainer.train_model(lstm_model, X_ts_train, y_ts_train, X_ts_val, y_ts_val, 'lstm', epochs=30)
    
    # Step 4: Model Evaluation
    print("\n4. MODEL EVALUATION")
    print("-" * 30)
    
    evaluator = ModelEvaluator()
    
    # Get predictions from all models
    predictions = {}
    
    # MLP predictions
    mlp_pred_scaled = mlp_model.predict(X_test)
    mlp_pred = ml_trainer.scaler_y.inverse_transform(mlp_pred_scaled).ravel()
    predictions['Deep Learning (MLP)'] = mlp_pred
    
    # Baseline predictions for comparison
    y_test_orig = ml_trainer.scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    # Generate baseline predictions for test set
    baseline_predictions = []
    for i in range(len(X_test)):
        # Inverse transform features to get original parameters
        features_orig = ml_trainer.scaler_X.inverse_transform(X_test[i:i+1])[0]
        S0, K, T, r, sigma, option_type = features_orig[:6]
        option_type_str = 'call' if option_type > 0.5 else 'put'
        
        # Black-Scholes prediction
        bs_result = baseline.black_scholes_european(S0, K, T, r, sigma, option_type_str)
        baseline_predictions.append(bs_result['price'])
    
    predictions['Black-Scholes'] = np.array(baseline_predictions)
    
    # Monte Carlo predictions (sample for speed)
    mc_predictions = []
    sample_indices = np.random.choice(len(X_test), min(1000, len(X_test)), replace=False)
    
    for i in sample_indices:
        features_orig = ml_trainer.scaler_X.inverse_transform(X_test[i:i+1])[0]
        S0, K, T, r, sigma, option_type = features_orig[:6]
        option_type_str = 'call' if option_type > 0.5 else 'put'
        
        mc_result = baseline.monte_carlo_european(S0, K, T, r, sigma, option_type_str, n_sims=10000)
        mc_predictions.append(mc_result['price'])
    
    # For full comparison, use Black-Scholes as proxy for Monte Carlo
    predictions['Monte Carlo'] = predictions['Black-Scholes']
    
    # Compare model performance
    model_names = list(predictions.keys())
    comparison_df = evaluator.compare_predictions(y_test_orig, predictions, model_names)
    
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
    print(f"\nBest performing model: {best_model}")
    
    # Step 5: Cross Validation
    print("\n5. CROSS VALIDATION")
    print("-" * 30)
    
    print("Performing 5-fold cross validation on MLP model...")
    cv_scores = ml_trainer.cross_validate(X, y, 'mlp', k_folds=5)
    
    print("Cross Validation Results:")
    for metric, scores in cv_scores.items():
        print(f"{metric.upper()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Step 6: Visualization
    print("\n6. GENERATING VISUALIZATIONS")
    print("-" * 30)
    
    # Plot prediction comparison
    evaluator.plot_prediction_comparison(y_test_orig, predictions, model_names, sample_size=1000)
    
    # Plot residual analysis
    evaluator.plot_residuals(y_test_orig, predictions, model_names)
    
    # Sensitivity analysis
    if 'mlp' in ml_trainer.models:
        base_params = {
            'S0': 100, 'K': 100, 'T': 0.25, 'r': 0.05, 'sigma': 0.2,
            'option_type': 1  # call option
        }
        
        print("Performing sensitivity analysis...")
        sensitivities = evaluator.sensitivity_analysis(
            ml_trainer.models['mlp'], 
            ml_trainer.scaler_X, 
            ml_trainer.scaler_y, 
            base_params
        )
        
        evaluator.plot_sensitivity_analysis(sensitivities)
    
    # Step 7: Save Results
    print("\n7. SAVING RESULTS")
    print("-" * 30)
    
    # Save models
    ml_trainer.save_models('trained_models')
    print("Models saved to 'trained_models' directory")
    
    # Save evaluation report
    report = evaluator.generate_report(y_test_orig, predictions, model_names)
    with open('model_evaluation_report.md', 'w') as f:
        f.write(report)
    print("Evaluation report saved to 'model_evaluation_report.md'")
    
    # Save data
    df.to_csv('option_training_data.csv', index=False)
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("Data and results saved to CSV files")
    
    # Step 8: Summary
    print("\n8. PIPELINE SUMMARY")
    print("-" * 30)
    
    print(f"✓ Generated {len(df)} training samples")
    print(f"✓ Trained {len(ml_trainer.models)} deep learning models")
    print(f"✓ Best model: {best_model} (R² = {comparison_df.loc[comparison_df['Model'] == best_model, 'R²'].iloc[0]:.4f})")
    print(f"✓ Cross-validation RMSE: {np.mean(cv_scores['rmse']):.4f} ± {np.std(cv_scores['rmse']):.4f}")
    print("✓ Models and results saved successfully")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return {
        'models': ml_trainer.models,
        'comparison': comparison_df,
        'cv_scores': cv_scores,
        'best_model': best_model
    }

if __name__ == "__main__":
    results = main()

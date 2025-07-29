import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import talib
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced feature engineering for option pricing with financial domain knowledge"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        self.pca_transformers = {}
        self.feature_names = []
        
    def create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sophisticated market-based features
        """
        features = df.copy()
        
        # Basic option features
        features['moneyness'] = features['S0'] / features['K']
        features['log_moneyness'] = np.log(features['moneyness'])
        features['time_to_expiry'] = features['T']
        features['sqrt_time'] = np.sqrt(features['T'])
        features['vol_time'] = features['sigma'] * np.sqrt(features['T'])
        
        # Advanced moneyness features
        features['moneyness_squared'] = features['moneyness'] ** 2
        features['moneyness_cubed'] = features['moneyness'] ** 3
        features['inverse_moneyness'] = 1 / features['moneyness']
        
        # Volatility features
        features['vol_squared'] = features['sigma'] ** 2
        features['vol_cubed'] = features['sigma'] ** 3
        features['log_vol'] = np.log(features['sigma'])
        features['vol_skew'] = features['sigma'] * features['moneyness']
        
        # Interest rate features
        features['rate_time'] = features['r'] * features['T']
        features['discount_factor'] = np.exp(-features['r'] * features['T'])
        features['forward_price'] = features['S0'] * np.exp(features['r'] * features['T'])
        features['forward_moneyness'] = features['forward_price'] / features['K']
        
        # Black-Scholes d1 and d2
        features['d1'] = (np.log(features['S0'] / features['K']) + 
                         (features['r'] + 0.5 * features['sigma']**2) * features['T']) / \
                        (features['sigma'] * np.sqrt(features['T']))
        features['d2'] = features['d1'] - features['sigma'] * np.sqrt(features['T'])
        
        # Greeks approximations as features
        features['delta_approx'] = stats.norm.cdf(features['d1'])
        features['gamma_approx'] = stats.norm.pdf(features['d1']) / \
                                  (features['S0'] * features['sigma'] * np.sqrt(features['T']))
        features['vega_approx'] = features['S0'] * stats.norm.pdf(features['d1']) * np.sqrt(features['T'])
        features['theta_approx'] = -(features['S0'] * stats.norm.pdf(features['d1']) * features['sigma'] / 
                                    (2 * np.sqrt(features['T'])) + 
                                    features['r'] * features['K'] * np.exp(-features['r'] * features['T']) * 
                                    stats.norm.cdf(features['d2']))
        
        # Risk-neutral probability features
        features['risk_neutral_prob'] = stats.norm.cdf(features['d2'])
        features['prob_itm'] = np.where(features['option_type'] == 1, 
                                       features['risk_neutral_prob'],
                                       1 - features['risk_neutral_prob'])
        
        # Volatility smile features (synthetic)
        features['vol_smile_skew'] = features['sigma'] * (1 + 0.1 * (features['moneyness'] - 1))
        features['vol_smile_convexity'] = features['sigma'] * (1 + 0.05 * (features['moneyness'] - 1)**2)
        
        # Time decay features
        features['time_decay_linear'] = 1 - features['T']
        features['time_decay_exponential'] = np.exp(-features['T'])
        features['time_decay_sqrt'] = 1 - np.sqrt(features['T'])
        
        return features
    
    def create_technical_indicators(self, price_series: np.ndarray, 
                                  volume_series: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Create technical indicators from price series
        """
        indicators = {}
        
        if len(price_series) < 50:
            # If series too short, create synthetic indicators
            indicators['rsi'] = np.full(len(price_series), 50.0)
            indicators['macd'] = np.zeros(len(price_series))
            indicators['bb_upper'] = price_series * 1.02
            indicators['bb_lower'] = price_series * 0.98
            indicators['atr'] = np.full(len(price_series), 0.02)
            return indicators
        
        # RSI
        try:
            indicators['rsi'] = talib.RSI(price_series.astype(float), timeperiod=14)
        except:
            indicators['rsi'] = np.full(len(price_series), 50.0)
        
        # MACD
        try:
            macd, macd_signal, macd_hist = talib.MACD(price_series.astype(float))
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_hist
        except:
            indicators['macd'] = np.zeros(len(price_series))
            indicators['macd_signal'] = np.zeros(len(price_series))
            indicators['macd_histogram'] = np.zeros(len(price_series))
        
        # Bollinger Bands
        try:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(price_series.astype(float))
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
        except:
            indicators['bb_upper'] = price_series * 1.02
            indicators['bb_middle'] = price_series
            indicators['bb_lower'] = price_series * 0.98
            indicators['bb_width'] = np.full(len(price_series), 0.04)
        
        # ATR (Average True Range)
        try:
            high = price_series * 1.01
            low = price_series * 0.99
            close = price_series
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
        except:
            indicators['atr'] = np.full(len(price_series), 0.02)
        
        # Fill NaN values
        for key, value in indicators.items():
            indicators[key] = np.nan_to_num(value, nan=np.nanmean(value) if not np.isnan(value).all() else 0.0)
        
        return indicators
    
    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market regime features using volatility clustering and trend analysis
        """
        features = df.copy()
        
        # Volatility regimes
        vol_quantiles = np.quantile(features['sigma'], [0.33, 0.67])
        features['vol_regime_low'] = (features['sigma'] <= vol_quantiles[0]).astype(int)
        features['vol_regime_medium'] = ((features['sigma'] > vol_quantiles[0]) & 
                                        (features['sigma'] <= vol_quantiles[1])).astype(int)
        features['vol_regime_high'] = (features['sigma'] > vol_quantiles[1]).astype(int)
        
        # Interest rate regimes
        rate_quantiles = np.quantile(features['r'], [0.33, 0.67])
        features['rate_regime_low'] = (features['r'] <= rate_quantiles[0]).astype(int)
        features['rate_regime_medium'] = ((features['r'] > rate_quantiles[0]) & 
                                         (features['r'] <= rate_quantiles[1])).astype(int)
        features['rate_regime_high'] = (features['r'] > rate_quantiles[1]).astype(int)
        
        # Moneyness regimes
        features['deep_otm'] = (features['moneyness'] < 0.9).astype(int)
        features['otm'] = ((features['moneyness'] >= 0.9) & (features['moneyness'] < 0.95)).astype(int)
        features['near_atm'] = ((features['moneyness'] >= 0.95) & (features['moneyness'] <= 1.05)).astype(int)
        features['itm'] = ((features['moneyness'] > 1.05) & (features['moneyness'] <= 1.1)).astype(int)
        features['deep_itm'] = (features['moneyness'] > 1.1).astype(int)
        
        # Time to expiry regimes
        features['short_term'] = (features['T'] <= 0.25).astype(int)
        features['medium_term'] = ((features['T'] > 0.25) & (features['T'] <= 1.0)).astype(int)
        features['long_term'] = (features['T'] > 1.0).astype(int)
        
        return features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables
        """
        features = df.copy()
        
        # Key interaction terms
        features['vol_moneyness'] = features['sigma'] * features['moneyness']
        features['vol_time'] = features['sigma'] * features['T']
        features['moneyness_time'] = features['moneyness'] * features['T']
        features['rate_time'] = features['r'] * features['T']
        features['vol_rate'] = features['sigma'] * features['r']
        
        # Higher order interactions
        features['vol_moneyness_time'] = features['sigma'] * features['moneyness'] * features['T']
        features['vol_squared_time'] = features['sigma']**2 * features['T']
        features['moneyness_squared_vol'] = features['moneyness']**2 * features['sigma']
        
        # Option type interactions
        features['call_moneyness'] = features['option_type'] * features['moneyness']
        features['call_vol'] = features['option_type'] * features['sigma']
        features['call_time'] = features['option_type'] * features['T']
        
        return features
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for key variables
        """
        features = df.copy()
        
        key_vars = ['moneyness', 'sigma', 'T', 'r']
        
        for var in key_vars:
            if var in features.columns:
                for d in range(2, degree + 1):
                    features[f'{var}_poly_{d}'] = features[var] ** d
        
        return features
    
    def apply_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                              method: str = 'univariate', k: int = 50) -> Tuple[np.ndarray, List[int]]:
        """
        Apply feature selection to reduce dimensionality
        """
        if method == 'univariate':
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_features = selector.get_support(indices=True)
            self.feature_selectors['univariate'] = selector
            
        elif method == 'pca':
            pca = PCA(n_components=min(k, X.shape[1]), random_state=42)
            X_selected = pca.fit_transform(X)
            selected_features = list(range(X_selected.shape[1]))
            self.pca_transformers['pca'] = pca
            
        else:
            X_selected = X
            selected_features = list(range(X.shape[1]))
        
        return X_selected, selected_features
    
    def engineer_features(self, df: pd.DataFrame, include_technical: bool = False,
                         include_regimes: bool = True, include_interactions: bool = True,
                         polynomial_degree: int = 2) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering pipeline
        """
        print("Starting advanced feature engineering...")
        
        # Start with market features
        features = self.create_market_features(df)
        print(f"Created market features: {features.shape[1]} features")
        
        # Add regime features
        if include_regimes:
            features = self.create_regime_features(features)
            print(f"Added regime features: {features.shape[1]} features")
        
        # Add interaction features
        if include_interactions:
            features = self.create_interaction_features(features)
            print(f"Added interaction features: {features.shape[1]} features")
        
        # Add polynomial features
        if polynomial_degree > 1:
            features = self.create_polynomial_features(features, polynomial_degree)
            print(f"Added polynomial features: {features.shape[1]} features")
        
        # Add technical indicators if price series available
        if include_technical and 'price_series' in df.columns:
            for idx, price_series in enumerate(df['price_series']):
                if isinstance(price_series, (list, np.ndarray)) and len(price_series) > 10:
                    indicators = self.create_technical_indicators(np.array(price_series))
                    for name, values in indicators.items():
                        features.loc[idx, f'tech_{name}'] = values[-1]  # Use latest value
        
        # Remove any infinite or extremely large values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        print(f"Final feature set: {features.shape[1]} features")
        return features
    
    def scale_features(self, X: np.ndarray, method: str = 'robust') -> np.ndarray:
        """
        Scale features using robust methods suitable for financial data
        """
        if method == 'robust':
            scaler = RobustScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            return X
        
        X_scaled = scaler.fit_transform(X)
        self.scalers[method] = scaler
        return X_scaled
    
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted scalers and selectors
        """
        # Apply same feature engineering
        features = self.engineer_features(df)
        
        # Select same features as training
        if hasattr(self, 'feature_names'):
            missing_cols = set(self.feature_names) - set(features.columns)
            for col in missing_cols:
                features[col] = 0  # Add missing columns with default values
            features = features[self.feature_names]
        
        # Apply scaling
        X = features.values
        if 'robust' in self.scalers:
            X = self.scalers['robust'].transform(X)
        
        # Apply feature selection
        if 'univariate' in self.feature_selectors:
            X = self.feature_selectors['univariate'].transform(X)
        elif 'pca' in self.pca_transformers:
            X = self.pca_transformers['pca'].transform(X)
        
        return X

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'S0': np.random.uniform(80, 120, n_samples),
        'K': np.random.uniform(80, 120, n_samples),
        'T': np.random.uniform(0.1, 2.0, n_samples),
        'r': np.random.uniform(0.01, 0.1, n_samples),
        'sigma': np.random.uniform(0.1, 0.5, n_samples),
        'option_type': np.random.choice([0, 1], n_samples),
        'target_price': np.random.uniform(1, 20, n_samples)
    })
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Engineer features
    engineered_features = engineer.engineer_features(
        sample_data, 
        include_regimes=True,
        include_interactions=True,
        polynomial_degree=2
    )
    
    print(f"Original features: {sample_data.shape[1]}")
    print(f"Engineered features: {engineered_features.shape[1]}")
    print(f"Feature names: {engineered_features.columns.tolist()[:10]}...")  # Show first 10
    
    # Scale features
    X = engineered_features.drop('target_price', axis=1).values
    y = engineered_features['target_price'].values
    
    X_scaled = engineer.scale_features(X, method='robust')
    print(f"Scaled features shape: {X_scaled.shape}")
    
    # Apply feature selection
    X_selected, selected_indices = engineer.apply_feature_selection(X_scaled, y, method='univariate', k=30)
    print(f"Selected features shape: {X_selected.shape}")
    print(f"Selected feature indices: {selected_indices[:10]}...")  # Show first 10

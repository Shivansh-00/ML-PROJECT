import numpy as np
import pandas as pd
from typing import Tuple, Dict
import matplotlib.pyplot as plt

class OptionDataGenerator:
    """Generate synthetic option pricing data using Monte Carlo simulation"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def geometric_brownian_motion(self, S0: float, mu: float, sigma: float, 
                                T: float, dt: float, n_paths: int) -> np.ndarray:
        """
        Generate asset price paths using Geometric Brownian Motion
        
        Args:
            S0: Initial stock price
            mu: Drift (risk-free rate)
            sigma: Volatility
            T: Time to maturity
            dt: Time step
            n_paths: Number of simulation paths
        
        Returns:
            Array of price paths [n_paths, n_steps]
        """
        n_steps = int(T / dt)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for t in range(1, n_steps + 1):
            Z = np.random.standard_normal(n_paths)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        return paths
    
    def monte_carlo_option_price(self, S0: float, K: float, T: float, r: float, 
                               sigma: float, option_type: str, n_simulations: int = 100000) -> float:
        """
        Price options using Monte Carlo simulation
        
        Args:
            S0: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            n_simulations: Number of Monte Carlo simulations
        
        Returns:
            Option price
        """
        # Generate final stock prices
        Z = np.random.standard_normal(n_simulations)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:  # put
            payoffs = np.maximum(K - ST, 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        return option_price
    
    def black_scholes_price(self, S0: float, K: float, T: float, r: float, 
                          sigma: float, option_type: str) -> float:
        """
        Calculate Black-Scholes option price
        
        Args:
            S0: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        
        Returns:
            Black-Scholes option price
        """
        from scipy.stats import norm
        
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        
        return price
    
    def generate_training_data(self, n_samples: int = 50000) -> pd.DataFrame:
        """
        Generate training dataset with various option parameters
        
        Args:
            n_samples: Number of training samples to generate
        
        Returns:
            DataFrame with features and target prices
        """
        data = []
        
        for _ in range(n_samples):
            # Random parameter ranges
            S0 = np.random.uniform(50, 150)  # Stock price
            K = np.random.uniform(50, 150)   # Strike price
            T = np.random.uniform(0.1, 2.0)  # Time to maturity (years)
            r = np.random.uniform(0.01, 0.1) # Risk-free rate
            sigma = np.random.uniform(0.1, 0.5) # Volatility
            option_type = np.random.choice(['call', 'put'])
            
            # Calculate prices using different methods
            mc_price = self.monte_carlo_option_price(S0, K, T, r, sigma, option_type, 10000)
            bs_price = self.black_scholes_price(S0, K, T, r, sigma, option_type)
            
            # Additional features
            moneyness = S0 / K
            time_value = T
            
            data.append({
                'S0': S0,
                'K': K,
                'T': T,
                'r': r,
                'sigma': sigma,
                'option_type': 1 if option_type == 'call' else 0,
                'moneyness': moneyness,
                'time_value': time_value,
                'mc_price': mc_price,
                'bs_price': bs_price,
                'target_price': mc_price  # Use MC as ground truth
            })
        
        return pd.DataFrame(data)
    
    def generate_time_series_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time series data for LSTM training
        
        Args:
            n_samples: Number of time series samples
        
        Returns:
            Tuple of (features, targets) for time series modeling
        """
        sequence_length = 30
        features = []
        targets = []
        
        for _ in range(n_samples):
            # Generate price path
            S0 = np.random.uniform(80, 120)
            mu = np.random.uniform(0.05, 0.15)
            sigma = np.random.uniform(0.15, 0.35)
            T = 1.0
            dt = T / sequence_length
            
            path = self.geometric_brownian_motion(S0, mu, sigma, T, dt, 1)[0]
            
            # Option parameters
            K = np.random.uniform(80, 120)
            r = np.random.uniform(0.02, 0.08)
            
            # Create sequence features
            sequence_features = []
            for i in range(sequence_length):
                remaining_time = T - (i * dt)
                if remaining_time > 0:
                    option_price = self.black_scholes_price(path[i], K, remaining_time, r, sigma, 'call')
                    sequence_features.append([path[i], K, remaining_time, r, sigma, option_price])
                else:
                    sequence_features.append([path[i], K, 0.01, r, sigma, max(path[i] - K, 0)])
            
            features.append(sequence_features)
            # Target is final option value
            final_price = max(path[-1] - K, 0)
            targets.append(final_price)
        
        return np.array(features), np.array(targets)

# Generate and save data
if __name__ == "__main__":
    generator = OptionDataGenerator()
    
    print("Generating training data...")
    df = generator.generate_training_data(50000)
    print(f"Generated {len(df)} samples")
    print(df.head())
    
    print("\nGenerating time series data...")
    X_ts, y_ts = generator.generate_time_series_data(5000)
    print(f"Generated time series data: {X_ts.shape}, {y_ts.shape}")
    
    # Save data
    df.to_csv('option_training_data.csv', index=False)
    np.save('time_series_features.npy', X_ts)
    np.save('time_series_targets.npy', y_ts)
    
    print("Data generation complete!")

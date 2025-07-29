import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Tuple
import matplotlib.pyplot as plt

class BaselineModels:
    """Implementation of baseline option pricing models"""
    
    def __init__(self):
        pass
    
    def black_scholes_european(self, S: float, K: float, T: float, r: float, 
                             sigma: float, option_type: str) -> Dict[str, float]:
        """
        Black-Scholes formula for European options
        
        Returns:
            Dictionary with price and Greeks
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
        # Common Greeks
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type.lower() == 'call' else norm.cdf(-d2))
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Per day
            'vega': vega / 100,    # Per 1% volatility change
            'rho': rho / 100       # Per 1% interest rate change
        }
    
    def monte_carlo_european(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str, n_sims: int = 100000) -> Dict[str, float]:
        """
        Monte Carlo pricing for European options
        """
        # Generate random paths
        Z = np.random.standard_normal(n_sims)
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Price and confidence interval
        discounted_payoffs = np.exp(-r * T) * payoffs
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_sims)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error)
        }
    
    def monte_carlo_american(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str, n_steps: int = 100, 
                           n_sims: int = 50000) -> Dict[str, float]:
        """
        Monte Carlo pricing for American options using Longstaff-Schwartz method
        """
        dt = T / n_steps
        
        # Generate price paths
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = S
        
        for t in range(1, n_steps + 1):
            Z = np.random.standard_normal(n_sims)
            paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        # Calculate intrinsic values
        if option_type.lower() == 'call':
            intrinsic = np.maximum(paths - K, 0)
        else:
            intrinsic = np.maximum(K - paths, 0)
        
        # Backward induction for American option
        cash_flows = intrinsic[:, -1]  # Final payoffs
        
        for t in range(n_steps - 1, 0, -1):
            # In-the-money paths
            itm = intrinsic[:, t] > 0
            
            if np.sum(itm) > 0:
                # Regression to find continuation value
                X = paths[itm, t]
                Y = cash_flows[itm] * np.exp(-r * dt)
                
                # Simple polynomial regression (degree 2)
                if len(X) > 3:
                    A = np.column_stack([np.ones(len(X)), X, X**2])
                    try:
                        coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
                        continuation_value = A @ coeffs
                        
                        # Exercise decision
                        exercise = intrinsic[itm, t] > continuation_value
                        cash_flows[itm] = np.where(exercise, intrinsic[itm, t], 
                                                 cash_flows[itm] * np.exp(-r * dt))
                    except:
                        cash_flows[itm] *= np.exp(-r * dt)
                else:
                    cash_flows[itm] *= np.exp(-r * dt)
            
            # Update cash flows for out-of-the-money
            otm = ~itm
            cash_flows[otm] *= np.exp(-r * dt)
        
        price = np.mean(cash_flows)
        std_error = np.std(cash_flows) / np.sqrt(n_sims)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error)
        }
    
    def compare_models(self, S: float, K: float, T: float, r: float, 
                      sigma: float, option_type: str) -> pd.DataFrame:
        """
        Compare different pricing models
        """
        results = []
        
        # Black-Scholes (European only)
        bs_result = self.black_scholes_european(S, K, T, r, sigma, option_type)
        results.append({
            'Model': 'Black-Scholes',
            'Price': bs_result['price'],
            'Delta': bs_result['delta'],
            'Gamma': bs_result['gamma'],
            'Theta': bs_result['theta'],
            'Vega': bs_result['vega']
        })
        
        # Monte Carlo European
        mc_eur = self.monte_carlo_european(S, K, T, r, sigma, option_type)
        results.append({
            'Model': 'Monte Carlo European',
            'Price': mc_eur['price'],
            'Std Error': mc_eur['std_error'],
            'CI Lower': mc_eur['confidence_interval'][0],
            'CI Upper': mc_eur['confidence_interval'][1]
        })
        
        # Monte Carlo American
        mc_amer = self.monte_carlo_american(S, K, T, r, sigma, option_type)
        results.append({
            'Model': 'Monte Carlo American',
            'Price': mc_amer['price'],
            'Std Error': mc_amer['std_error'],
            'CI Lower': mc_amer['confidence_interval'][0],
            'CI Upper': mc_amer['confidence_interval'][1]
        })
        
        return pd.DataFrame(results)

# Test the models
if __name__ == "__main__":
    models = BaselineModels()
    
    # Test parameters
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
    
    print("Comparing pricing models for Call option:")
    print("S=$100, K=$100, T=0.25 years, r=5%, σ=20%")
    print()
    
    comparison = models.compare_models(S, K, T, r, sigma, 'call')
    print(comparison.to_string(index=False))
    
    print("\n" + "="*50)
    print("Comparing pricing models for Put option:")
    
    comparison_put = models.compare_models(S, K, T, r, sigma, 'put')
    print(comparison_put.to_string(index=False))

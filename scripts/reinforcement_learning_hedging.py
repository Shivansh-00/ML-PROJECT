import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from collections import deque
import random
from dataclasses import dataclass

@dataclass
class MarketState:
    """Market state representation for RL environment"""
    stock_price: float
    option_price: float
    time_to_expiry: float
    volatility: float
    risk_free_rate: float
    delta: float
    gamma: float
    theta: float
    vega: float
    portfolio_value: float
    hedge_ratio: float
    cash_position: float
    stock_position: float

class OptionHedgingEnvironment(gym.Env):
    """
    Reinforcement Learning environment for option hedging strategies
    """
    
    def __init__(self, initial_stock_price: float = 100.0, strike_price: float = 100.0,
                 initial_time_to_expiry: float = 0.25, volatility: float = 0.2,
                 risk_free_rate: float = 0.05, transaction_cost: float = 0.001):
        
        super().__init__()
        
        # Market parameters
        self.initial_stock_price = initial_stock_price
        self.strike_price = strike_price
        self.initial_time_to_expiry = initial_time_to_expiry
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        
        # Environment parameters
        self.dt = 1/252  # Daily time steps
        self.max_steps = int(initial_time_to_expiry / self.dt)
        
        # Action space: hedge ratio adjustment (-1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: market state features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(13,),  # 13 features in MarketState
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.stock_price = self.initial_stock_price
        self.time_to_expiry = self.initial_time_to_expiry
        
        # Initialize portfolio
        self.option_position = -1.0  # Short one option
        self.stock_position = 0.0
        self.cash_position = 0.0
        
        # Calculate initial option price and Greeks
        self.option_price = self._black_scholes_price()
        self.delta, self.gamma, self.theta, self.vega = self._calculate_greeks()
        
        # Initialize portfolio value
        self.portfolio_value = (self.option_position * self.option_price + 
                               self.stock_position * self.stock_price + 
                               self.cash_position)
        
        self.hedge_ratio = 0.0
        self.total_transaction_costs = 0.0
        self.pnl_history = []
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        # Extract action
        hedge_adjustment = action[0]
        
        # Store previous values
        prev_portfolio_value = self.portfolio_value
        prev_stock_position = self.stock_position
        
        # Update hedge ratio
        new_hedge_ratio = np.clip(self.hedge_ratio + hedge_adjustment, -2.0, 2.0)
        
        # Calculate required stock position change
        required_stock_position = new_hedge_ratio
        stock_trade = required_stock_position - self.stock_position
        
        # Apply transaction costs
        transaction_cost = abs(stock_trade) * self.stock_price * self.transaction_cost
        self.total_transaction_costs += transaction_cost
        
        # Update positions
        self.stock_position = required_stock_position
        self.cash_position -= stock_trade * self.stock_price + transaction_cost
        self.hedge_ratio = new_hedge_ratio
        
        # Simulate market movement
        self._simulate_market_step()
        
        # Update option price and Greeks
        self.option_price = self._black_scholes_price()
        self.delta, self.gamma, self.theta, self.vega = self._calculate_greeks()
        
        # Calculate new portfolio value
        new_portfolio_value = (self.option_position * self.option_price + 
                              self.stock_position * self.stock_price + 
                              self.cash_position)
        
        # Calculate reward (negative of PnL change to minimize hedging error)
        pnl_change = new_portfolio_value - prev_portfolio_value
        self.pnl_history.append(pnl_change)
        
        # Reward is negative of absolute PnL change (we want to minimize variance)
        reward = -abs(pnl_change) - transaction_cost * 10  # Penalize transaction costs
        
        self.portfolio_value = new_portfolio_value
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or self.time_to_expiry <= 0
        
        # Additional info
        info = {
            'pnl_change': pnl_change,
            'transaction_cost': transaction_cost,
            'total_transaction_costs': self.total_transaction_costs,
            'portfolio_value': self.portfolio_value,
            'hedge_ratio': self.hedge_ratio,
            'delta': self.delta
        }
        
        return self._get_observation(), reward, done, info
    
    def _simulate_market_step(self):
        """Simulate one step of market evolution using GBM"""
        # Generate random shock
        dW = np.random.normal(0, np.sqrt(self.dt))
        
        # Update stock price using GBM
        self.stock_price *= np.exp((self.risk_free_rate - 0.5 * self.volatility**2) * self.dt + 
                                  self.volatility * dW)
        
        # Update time to expiry
        self.time_to_expiry = max(0, self.time_to_expiry - self.dt)
    
    def _black_scholes_price(self) -> float:
        """Calculate Black-Scholes option price"""
        if self.time_to_expiry <= 0:
            return max(self.stock_price - self.strike_price, 0)  # Intrinsic value
        
        from scipy.stats import norm
        
        d1 = (np.log(self.stock_price / self.strike_price) + 
              (self.risk_free_rate + 0.5 * self.volatility**2) * self.time_to_expiry) / \
             (self.volatility * np.sqrt(self.time_to_expiry))
        d2 = d1 - self.volatility * np.sqrt(self.time_to_expiry)
        
        call_price = (self.stock_price * norm.cdf(d1) - 
                     self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(d2))
        
        return call_price
    
    def _calculate_greeks(self) -> Tuple[float, float, float, float]:
        """Calculate option Greeks"""
        if self.time_to_expiry <= 0:
            delta = 1.0 if self.stock_price > self.strike_price else 0.0
            return delta, 0.0, 0.0, 0.0
        
        from scipy.stats import norm
        
        d1 = (np.log(self.stock_price / self.strike_price) + 
              (self.risk_free_rate + 0.5 * self.volatility**2) * self.time_to_expiry) / \
             (self.volatility * np.sqrt(self.time_to_expiry))
        d2 = d1 - self.volatility * np.sqrt(self.time_to_expiry)
        
        # Delta
        delta = norm.cdf(d1)
        
        # Gamma
        gamma = norm.pdf(d1) / (self.stock_price * self.volatility * np.sqrt(self.time_to_expiry))
        
        # Theta
        theta = (-(self.stock_price * norm.pdf(d1) * self.volatility) / (2 * np.sqrt(self.time_to_expiry)) -
                self.risk_free_rate * self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(d2))
        theta /= 365  # Per day
        
        # Vega
        vega = self.stock_price * norm.pdf(d1) * np.sqrt(self.time_to_expiry) / 100
        
        return delta, gamma, theta, vega
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        return np.array([
            self.stock_price / 100.0,  # Normalized stock price
            self.option_price / 10.0,  # Normalized option price
            self.time_to_expiry * 4,   # Normalized time to expiry
            self.volatility * 5,       # Normalized volatility
            self.risk_free_rate * 20,  # Normalized risk-free rate
            self.delta,
            self.gamma * 100,          # Scaled gamma
            self.theta * 100,          # Scaled theta
            self.vega / 10,            # Scaled vega
            self.portfolio_value / 100, # Normalized portfolio value
            self.hedge_ratio / 2,      # Normalized hedge ratio
            self.cash_position / 100,  # Normalized cash position
            self.stock_position / 2    # Normalized stock position
        ], dtype=np.float32)

class PPOAgent:
    """
    Proximal Policy Optimization agent for option hedging
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # Hyperparameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.target_kl = 0.01
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        
        # Networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=lr)
        
        # Memory
        self.memory = []
    
    def _build_actor(self) -> keras.Model:
        """Build actor network (policy)"""
        inputs = layers.Input(shape=(self.state_dim,))
        
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        # Output mean and log_std for continuous actions
        mean = layers.Dense(self.action_dim, activation='tanh')(x)
        log_std = layers.Dense(self.action_dim, activation='linear')(x)
        log_std = tf.clip_by_value(log_std, -20, 2)
        
        model = keras.Model(inputs=inputs, outputs=[mean, log_std])
        return model
    
    def _build_critic(self) -> keras.Model:
        """Build critic network (value function)"""
        inputs = layers.Input(shape=(self.state_dim,))
        
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        value = layers.Dense(1, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=value)
        return model
    
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float, float]:
        """Get action from policy"""
        state = tf.expand_dims(state, 0)
        mean, log_std = self.actor(state)
        std = tf.exp(log_std)
        
        if training:
            # Sample from policy distribution
            action = tf.random.normal(tf.shape(mean), mean, std)
        else:
            # Use mean action for evaluation
            action = mean
        
        action = tf.clip_by_value(action, -1.0, 1.0)
        
        # Calculate log probability
        log_prob = -0.5 * tf.reduce_sum(
            tf.square((action - mean) / std) + 2 * log_std + np.log(2 * np.pi), axis=1
        )
        
        return action.numpy()[0], log_prob.numpy()[0], tf.reduce_mean(std).numpy()
    
    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate from critic"""
        state = tf.expand_dims(state, 0)
        value = self.critic(state)
        return value.numpy()[0, 0]
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, log_prob: float, value: float):
        """Store transition in memory"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   next_values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def update(self, batch_size: int = 64, epochs: int = 10):
        """Update policy using PPO"""
        if len(self.memory) < batch_size:
            return
        
        # Prepare data
        states = np.array([m['state'] for m in self.memory])
        actions = np.array([m['action'] for m in self.memory])
        rewards = [m['reward'] for m in self.memory]
        next_states = np.array([m['next_state'] for m in self.memory])
        dones = [m['done'] for m in self.memory]
        old_log_probs = np.array([m['log_prob'] for m in self.memory])
        values = [m['value'] for m in self.memory]
        
        # Calculate next values
        next_values = [self.get_value(ns) for ns in next_states]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        advantages = np.array(advantages)
        returns = np.array(returns)
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Convert to tensors
        states = tf.constant(states, dtype=tf.float32)
        actions = tf.constant(actions, dtype=tf.float32)
        old_log_probs = tf.constant(old_log_probs, dtype=tf.float32)
        advantages = tf.constant(advantages, dtype=tf.float32)
        returns = tf.constant(returns, dtype=tf.float32)
        
        # Update for multiple epochs
        for epoch in range(epochs):
            # Shuffle data
            indices = tf.random.shuffle(tf.range(tf.shape(states)[0]))
            states_shuffled = tf.gather(states, indices)
            actions_shuffled = tf.gather(actions, indices)
            old_log_probs_shuffled = tf.gather(old_log_probs, indices)
            advantages_shuffled = tf.gather(advantages, indices)
            returns_shuffled = tf.gather(returns, indices)
            
            # Mini-batch updates
            for i in range(0, len(states), batch_size):
                batch_states = states_shuffled[i:i+batch_size]
                batch_actions = actions_shuffled[i:i+batch_size]
                batch_old_log_probs = old_log_probs_shuffled[i:i+batch_size]
                batch_advantages = advantages_shuffled[i:i+batch_size]
                batch_returns = returns_shuffled[i:i+batch_size]
                
                # Update actor
                with tf.GradientTape() as tape:
                    mean, log_std = self.actor(batch_states)
                    std = tf.exp(log_std)
                    
                    # Calculate new log probabilities
                    new_log_probs = -0.5 * tf.reduce_sum(
                        tf.square((batch_actions - mean) / std) + 2 * log_std + np.log(2 * np.pi), axis=1
                    )
                    
                    # Calculate ratio
                    ratio = tf.exp(new_log_probs - batch_old_log_probs)
                    
                    # Calculate surrogate losses
                    surr1 = ratio * batch_advantages
                    surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                    
                    # Policy loss
                    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                    
                    # Entropy bonus
                    entropy = tf.reduce_mean(0.5 * (tf.math.log(2 * np.pi * tf.square(std)) + 1))
                    
                    # Total actor loss
                    actor_loss = policy_loss - self.entropy_coef * entropy
                
                # Update actor
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                
                # Update critic
                with tf.GradientTape() as tape:
                    values_pred = tf.squeeze(self.critic(batch_states))
                    critic_loss = tf.reduce_mean(tf.square(batch_returns - values_pred))
                
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Clear memory
        self.memory = []

class HedgingTrainer:
    """
    Trainer for reinforcement learning hedging strategies
    """
    
    def __init__(self, env: OptionHedgingEnvironment):
        self.env = env
        self.agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0])
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []
        self.hedge_ratios = []
        self.transaction_costs = []
    
    def train(self, episodes: int = 1000, update_frequency: int = 20):
        """Train the hedging agent"""
        print(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_portfolio_values = []
            episode_hedge_ratios = []
            episode_transaction_costs = 0
            
            while True:
                # Get action from agent
                action, log_prob, _ = self.agent.get_action(state, training=True)
                value = self.agent.get_value(state)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done, log_prob, value)
                
                # Update metrics
                episode_reward += reward
                episode_length += 1
                episode_portfolio_values.append(info['portfolio_value'])
                episode_hedge_ratios.append(info['hedge_ratio'])
                episode_transaction_costs += info['transaction_cost']
                
                state = next_state
                
                if done:
                    break
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.portfolio_values.append(episode_portfolio_values)
            self.hedge_ratios.append(episode_hedge_ratios)
            self.transaction_costs.append(episode_transaction_costs)
            
            # Update agent
            if (episode + 1) % update_frequency == 0:
                self.agent.update()
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                avg_cost = np.mean(self.transaction_costs[-100:])
                
                print(f"Episode {episode + 1}/{episodes}")
                print(f"  Average Reward: {avg_reward:.4f}")
                print(f"  Average Length: {avg_length:.1f}")
                print(f"  Average Transaction Cost: {avg_cost:.6f}")
                print()
    
    def evaluate(self, episodes: int = 100) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        print(f"Evaluating agent for {episodes} episodes...")
        
        eval_rewards = []
        eval_portfolio_values = []
        eval_hedge_ratios = []
        eval_transaction_costs = []
        eval_pnl_variances = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_portfolio_values = []
            episode_hedge_ratios = []
            episode_transaction_costs = 0
            
            while True:
                # Get action (no exploration)
                action, _, _ = self.agent.get_action(state, training=False)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_portfolio_values.append(info['portfolio_value'])
                episode_hedge_ratios.append(info['hedge_ratio'])
                episode_transaction_costs += info['transaction_cost']
                
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_portfolio_values.append(episode_portfolio_values)
            eval_hedge_ratios.append(episode_hedge_ratios)
            eval_transaction_costs.append(episode_transaction_costs)
            
            # Calculate PnL variance
            pnl_changes = self.env.pnl_history
            eval_pnl_variances.append(np.var(pnl_changes))
        
        # Calculate evaluation metrics
        results = {
            'average_reward': np.mean(eval_rewards),
            'reward_std': np.std(eval_rewards),
            'average_transaction_cost': np.mean(eval_transaction_costs),
            'average_pnl_variance': np.mean(eval_pnl_variances),
            'portfolio_values': eval_portfolio_values,
            'hedge_ratios': eval_hedge_ratios
        }
        
        return results
    
    def plot_training_progress(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Hedging Training Progress', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Moving average of rewards
        window = 50
        if len(self.episode_rewards) >= window:
            moving_avg = pd.Series(self.episode_rewards).rolling(window=window).mean()
            axes[0, 0].plot(moving_avg, color='red', label=f'{window}-episode MA')
            axes[0, 0].legend()
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Transaction costs
        axes[1, 0].plot(self.transaction_costs)
        axes[1, 0].set_title('Transaction Costs per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Transaction Cost')
        
        # Sample hedge ratio evolution
        if self.hedge_ratios:
            sample_episode = min(len(self.hedge_ratios) - 1, 100)
            axes[1, 1].plot(self.hedge_ratios[sample_episode])
            axes[1, 1].set_title(f'Hedge Ratio Evolution (Episode {sample_episode})')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Hedge Ratio')
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_delta_hedging(self, episodes: int = 100):
        """Compare RL agent with traditional delta hedging"""
        print("Comparing RL agent with delta hedging...")
        
        # Evaluate RL agent
        rl_results = self.evaluate(episodes)
        
        # Evaluate delta hedging
        delta_rewards = []
        delta_portfolio_values = []
        delta_transaction_costs = []
        delta_pnl_variances = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_portfolio_values = []
            episode_transaction_costs = 0
            
            while True:
                # Delta hedging action (hedge ratio = delta)
                current_delta = self.env.delta
                current_hedge_ratio = self.env.hedge_ratio
                action = np.array([current_delta - current_hedge_ratio])
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_portfolio_values.append(info['portfolio_value'])
                episode_transaction_costs += info['transaction_cost']
                
                state = next_state
                
                if done:
                    break
            
            delta_rewards.append(episode_reward)
            delta_portfolio_values.append(episode_portfolio_values)
            delta_transaction_costs.append(episode_transaction_costs)
            
            pnl_changes = self.env.pnl_history
            delta_pnl_variances.append(np.var(pnl_changes))
        
        # Print comparison
        print("\nComparison Results:")
        print(f"RL Agent:")
        print(f"  Average Reward: {rl_results['average_reward']:.4f}")
        print(f"  Average Transaction Cost: {rl_results['average_transaction_cost']:.6f}")
        print(f"  Average PnL Variance: {rl_results['average_pnl_variance']:.6f}")
        
        print(f"\nDelta Hedging:")
        print(f"  Average Reward: {np.mean(delta_rewards):.4f}")
        print(f"  Average Transaction Cost: {np.mean(delta_transaction_costs):.6f}")
        print(f"  Average PnL Variance: {np.mean(delta_pnl_variances):.6f}")
        
        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Rewards comparison
        axes[0].boxplot([rl_results['average_reward']] * episodes, positions=[1], labels=['RL Agent'])
        axes[0].boxplot(delta_rewards, positions=[2], labels=['Delta Hedging'])
        axes[0].set_title('Reward Comparison')
        axes[0].set_ylabel('Episode Reward')
        
        # Transaction costs comparison
        axes[1].boxplot([rl_results['average_transaction_cost']] * episodes, positions=[1])
        axes[1].boxplot(delta_transaction_costs, positions=[2])
        axes[1].set_title('Transaction Cost Comparison')
        axes[1].set_ylabel('Total Transaction Cost')
        axes[1].set_xticklabels(['RL Agent', 'Delta Hedging'])
        
        # PnL variance comparison
        axes[2].boxplot([rl_results['average_pnl_variance']] * episodes, positions=[1])
        axes[2].boxplot(delta_pnl_variances, positions=[2])
        axes[2].set_title('PnL Variance Comparison')
        axes[2].set_ylabel('PnL Variance')
        axes[2].set_xticklabels(['RL Agent', 'Delta Hedging'])
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create environment
    env = OptionHedgingEnvironment(
        initial_stock_price=100.0,
        strike_price=100.0,
        initial_time_to_expiry=0.25,
        volatility=0.2,
        risk_free_rate=0.05,
        transaction_cost=0.001
    )
    
    # Create trainer
    trainer = HedgingTrainer(env)
    
    # Train agent
    print("Training RL hedging agent...")
    trainer.train(episodes=500, update_frequency=20)
    
    # Plot training progress
    trainer.plot_training_progress()
    
    # Evaluate agent
    evaluation_results = trainer.evaluate(episodes=100)
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {evaluation_results['average_reward']:.4f}")
    print(f"Reward Std: {evaluation_results['reward_std']:.4f}")
    print(f"Average Transaction Cost: {evaluation_results['average_transaction_cost']:.6f}")
    print(f"Average PnL Variance: {evaluation_results['average_pnl_variance']:.6f}")
    
    # Compare with delta hedging
    trainer.compare_with_delta_hedging(episodes=100)
    
    print("\nReinforcement learning hedging demonstration complete!")

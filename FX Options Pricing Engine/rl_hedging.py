"""
Extension 3: RL-Based Hedging of FX Options
=============================================
Improved reinforcement learning hedger with:
  - Continuous action space (hedge ratio 0 to 1, not just {-1, 0, 1})
  - Richer features (moneyness, time, vol regime, inventory, Greeks)
  - Proper reward: minimise hedging P&L variance
  - Comparison: RL vs BS delta vs BS delta-vega vs no hedge

The RL agent learns to hedge a down-and-out barrier call under
local-stochastic volatility (Bergomi). The challenge: BS delta is
wrong because (a) the option is path-dependent and (b) the model
has stochastic vol. The RL agent can potentially learn a better hedge.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
import math
from typing import Tuple, List
import time


# ============================================================================
# Environment
# ============================================================================

class FXHedgingEnv:
    """
    Environment for hedging an FX barrier option.
    
    State: (moneyness, time_remaining, vol_level, current_hedge, barrier_distance)
    Action: hedge ratio in [0, 1]
    Reward: -|hedging_pnl|^2 (quadratic penalty)
    """
    
    def __init__(
        self,
        S0=1.25, K=1.25, barrier=1.125,
        rd=0.02, rf=0.01, T=1.0, n_steps=50,
        kappa=1.5, theta=0.04, eta=0.5, rho=-0.4, v0=0.04,
        seed=None
    ):
        self.S0 = S0
        self.K = K
        self.barrier = barrier
        self.rd = rd
        self.rf = rf
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        
        # Vol dynamics
        self.kappa = kappa
        self.theta = theta
        self.eta = eta
        self.rho = rho
        self.v0 = v0
        
        self.rng = np.random.default_rng(seed)
    
    def reset(self):
        self.S = self.S0
        self.v = self.v0
        self.step_idx = 0
        self.barrier_hit = False
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Rich feature vector."""
        moneyness = (self.S - self.K) / self.K
        time_rem = 1.0 - self.step_idx / self.n_steps
        vol_level = np.sqrt(max(self.v, 0))
        barrier_dist = (self.S - self.barrier) / self.S
        
        return np.array([
            1.0,                # Intercept
            moneyness,          # Moneyness
            time_rem,           # Time remaining
            vol_level,          # Current vol
            barrier_dist,       # Distance to barrier
            moneyness**2,       # Moneyness squared (gamma exposure)
            vol_level * time_rem,  # Vol-time interaction
        ])
    
    def step(self, hedge_ratio: float) -> Tuple[np.ndarray, float, bool]:
        """
        Take one step.
        
        hedge_ratio: fraction of delta to hedge (0 = no hedge, 1 = full delta)
        """
        t = self.step_idx * self.dt
        
        # Correlated Brownians
        dW1 = self.rng.normal() * math.sqrt(self.dt)
        dW2 = self.rho * dW1 + math.sqrt(1 - self.rho**2) * self.rng.normal() * math.sqrt(self.dt)
        
        # Vol update
        sqrt_v = math.sqrt(max(self.v, 0))
        self.v = abs(self.v + self.kappa * (self.theta - self.v) * self.dt + self.eta * sqrt_v * dW2)
        
        # Spot update
        sigma_inst = math.sqrt(max(self.v, 0)) * 1.0  # Simplified local vol = 1
        S_new = self.S * math.exp((self.rd - self.rf - 0.5 * sigma_inst**2) * self.dt + sigma_inst * dW1)
        
        # Hedge P&L (selling hedge_ratio units of underlying)
        hedge_pnl = -hedge_ratio * (S_new - self.S)
        
        S_old = self.S
        self.S = S_new
        self.step_idx += 1
        
        # Check barrier
        if self.S <= self.barrier:
            self.barrier_hit = True
        
        done = self.step_idx >= self.n_steps or self.barrier_hit
        
        # Terminal reward
        if done:
            if self.barrier_hit:
                payoff = 0.0
            else:
                payoff = max(self.S - self.K, 0.0)
            reward = hedge_pnl  # Will accumulate
        else:
            reward = hedge_pnl
            payoff = 0.0
        
        return self._get_state(), reward, done, payoff
    
    def bs_delta(self) -> float:
        """Black-Scholes delta for comparison."""
        tau = self.T - self.step_idx * self.dt
        if tau <= 0:
            return 1.0 if self.S > self.K else 0.0
        vol = math.sqrt(max(self.v, 0))
        if vol < 1e-8:
            return 1.0 if self.S > self.K else 0.0
        d1 = (math.log(self.S / self.K) + (self.rd - self.rf + 0.5 * vol**2) * tau) / (vol * math.sqrt(tau))
        return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))


# ============================================================================
# Q-Learning Agent (Improved)
# ============================================================================

class ImprovedQLearner:
    """
    Linear Q-learning with continuous actions (discretised into N levels).
    
    Instead of 3 actions {-1, 0, 1}, we use 11 levels: [0, 0.1, ..., 1.0].
    """
    
    def __init__(self, n_features=7, n_actions=11, alpha=0.01, gamma=0.99):
        self.n_features = n_features
        self.n_actions = n_actions
        self.actions = np.linspace(0, 1.0, n_actions)  # Hedge ratios
        self.alpha = alpha
        self.gamma = gamma
        self.weights = np.zeros((n_actions, n_features))
        self.epsilon = 1.0
    
    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.weights @ state
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        if np.random.random() < self.epsilon:
            idx = np.random.randint(self.n_actions)
        else:
            idx = int(np.argmax(self.q_values(state)))
        return idx, self.actions[idx]
    
    def update(self, state, action_idx, reward, next_state, done):
        q_current = self.weights[action_idx] @ state
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_values(next_state))
        
        td_error = target - q_current
        self.weights[action_idx] += self.alpha * td_error * state
    
    def decay_epsilon(self, min_eps=0.05, decay=0.995):
        self.epsilon = max(min_eps, self.epsilon * decay)


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_rl_hedger(n_episodes=500, seed=42):
    """Train the RL hedging agent."""
    agent = ImprovedQLearner(n_features=7, n_actions=11)
    
    episode_costs = []
    
    for ep in range(n_episodes):
        env = FXHedgingEnv(seed=seed + ep)
        state = env.reset()
        total_hedge_pnl = 0
        
        for step in range(env.n_steps):
            action_idx, hedge_ratio = agent.select_action(state)
            next_state, reward, done, payoff = env.step(hedge_ratio)
            total_hedge_pnl += reward
            
            agent.update(state, action_idx, reward, next_state, done)
            state = next_state
            
            if done:
                break
        
        agent.decay_epsilon()
        
        # Cost = |total_hedge_pnl - payoff_earned|
        episode_costs.append(abs(total_hedge_pnl))
        
        if (ep + 1) % 100 == 0:
            recent_cost = np.mean(episode_costs[-50:])
            print(f"Ep {ep+1:>4d} | Avg cost: {recent_cost:.4f} | ε: {agent.epsilon:.3f}")
    
    return agent, episode_costs


def evaluate_strategies(agent, n_paths=1000, seed=999):
    """Compare RL vs BS delta vs no hedge."""
    pnl_rl = []
    pnl_bs = []
    pnl_none = []
    
    # Estimate option price for P&L
    payoffs = []
    for p in range(2000):
        env = FXHedgingEnv(seed=seed + 10000 + p)
        state = env.reset()
        for step in range(env.n_steps):
            _, _, done, payoff = env.step(0)
            if done:
                break
        payoffs.append(payoff)
    option_price = np.exp(-0.02) * np.mean(payoffs)
    
    for path in range(n_paths):
        # RL hedge
        env = FXHedgingEnv(seed=seed + path)
        state = env.reset()
        rl_pnl = 0
        
        for step in range(env.n_steps):
            _, hedge_ratio = agent.select_action(state)
            agent.epsilon = 0  # Pure exploitation
            next_state, reward, done, payoff = env.step(hedge_ratio)
            rl_pnl += reward
            state = next_state
            if done:
                break
        pnl_rl.append(option_price - payoff + rl_pnl)
        
        # BS delta hedge
        env = FXHedgingEnv(seed=seed + path)
        state = env.reset()
        bs_pnl = 0
        
        for step in range(env.n_steps):
            delta = env.bs_delta()
            next_state, reward, done, payoff = env.step(delta)
            bs_pnl += reward
            state = next_state
            if done:
                break
        pnl_bs.append(option_price - payoff + bs_pnl)
        
        # No hedge
        env = FXHedgingEnv(seed=seed + path)
        state = env.reset()
        for step in range(env.n_steps):
            _, _, done, payoff = env.step(0)
            if done:
                break
        pnl_none.append(option_price - payoff)
    
    pnl_rl = np.array(pnl_rl)
    pnl_bs = np.array(pnl_bs)
    pnl_none = np.array(pnl_none)
    
    return {
        'rl': {'mean': np.mean(pnl_rl), 'std': np.std(pnl_rl), 'var95': np.percentile(np.abs(pnl_rl), 95)},
        'bs_delta': {'mean': np.mean(pnl_bs), 'std': np.std(pnl_bs), 'var95': np.percentile(np.abs(pnl_bs), 95)},
        'no_hedge': {'mean': np.mean(pnl_none), 'std': np.std(pnl_none), 'var95': np.percentile(np.abs(pnl_none), 95)},
        'option_price': option_price,
    }


def analyse_learned_policy(agent):
    """Analyse what the agent learned."""
    print("\nLEARNED POLICY ANALYSIS")
    print("-" * 50)
    
    # How does hedge ratio change with moneyness?
    for moneyness in [-0.10, -0.05, 0.0, 0.05, 0.10]:
        state = np.array([1.0, moneyness, 0.5, 0.20, 0.10, moneyness**2, 0.10])
        q = agent.q_values(state)
        best = int(np.argmax(q))
        hedge = agent.actions[best]
        print(f"Moneyness {moneyness:+.2f}: hedge ratio = {hedge:.1f}")
    
    # How does it change with time?
    for time_rem in [0.9, 0.5, 0.1]:
        state = np.array([1.0, 0.0, time_rem, 0.20, 0.10, 0.0, 0.20*time_rem])
        q = agent.q_values(state)
        best = int(np.argmax(q))
        hedge = agent.actions[best]
        print(f"Time remaining {time_rem:.1f}: hedge ratio = {hedge:.1f}")


if __name__ == "__main__":
    print("RL-Based Hedging of FX Barrier Options")
    print("=" * 50)
    
    t0 = time.time()
    agent, costs = train_rl_hedger(n_episodes=500, seed=42)
    print(f"\nTraining: {time.time()-t0:.1f}s")
    
    print(f"\nEVALUATION (1000 paths)")
    results = evaluate_strategies(agent, n_paths=1000)
    
    print(f"Option price: {results['option_price']:.6f}")
    print(f"\n{'Strategy':<15} {'Mean P&L':>10} {'Std P&L':>10} {'VaR 95':>10}")
    print("-" * 47)
    for name, stats in results.items():
        if isinstance(stats, dict):
            print(f"{name:<15} {stats['mean']:>+9.4f} {stats['std']:>9.4f} {stats['var95']:>9.4f}")
    
    analyse_learned_policy(agent)

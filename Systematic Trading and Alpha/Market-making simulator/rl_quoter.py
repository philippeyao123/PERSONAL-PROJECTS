"""
Extension 4: Reinforcement Learning Market-Maker
==================================================
A Q-learning agent that learns the optimal quoting strategy by
interacting with the simulated market environment.

State space:
    - Inventory level (discretised)
    - Volatility regime (low/high)
    - Time remaining (discretised)
    - Recent order imbalance

Action space:
    - Spread level (tight/medium/wide)
    - Inventory skew (buy-lean/neutral/sell-lean)

Reward:
    - Realised P&L per step
    - Penalty for large inventory (risk aversion)

The RL agent discovers that it should:
    - Widen spreads in high vol
    - Skew quotes to reduce inventory
    - Be more aggressive near end-of-day

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import time


# ============================================================================
# Environment
# ============================================================================

@dataclass
class MarketState:
    """Discretised market state for RL."""
    inventory_bin: int       # [-2, -1, 0, 1, 2] (5 bins)
    vol_regime: int          # 0=low, 1=high
    time_bin: int            # 0=early, 1=mid, 2=late (3 bins)
    imbalance_bin: int       # 0=sell, 1=neutral, 2=buy (3 bins)
    
    @property
    def index(self) -> int:
        """Flatten to single state index."""
        return (self.inventory_bin * 2 * 3 * 3 +
                self.vol_regime * 3 * 3 +
                self.time_bin * 3 +
                self.imbalance_bin)
    
    @staticmethod
    def n_states() -> int:
        return 5 * 2 * 3 * 3  # 90 states


@dataclass
class Action:
    """Discretised action space."""
    spread_level: int    # 0=tight, 1=medium, 2=wide
    skew_level: int      # 0=buy-lean, 1=neutral, 2=sell-lean
    
    @property
    def index(self) -> int:
        return self.spread_level * 3 + self.skew_level
    
    @staticmethod
    def n_actions() -> int:
        return 3 * 3  # 9 actions
    
    @staticmethod
    def from_index(idx: int) -> 'Action':
        return Action(spread_level=idx // 3, skew_level=idx % 3)


# Spread and skew mappings
SPREAD_MAP = {0: 0.002, 1: 0.005, 2: 0.012}     # Half-spread as fraction of mid
SKEW_MAP = {0: -0.002, 1: 0.0, 2: 0.002}          # Price adjustment


class MMEnvironment:
    """
    Market-making environment for RL training.
    
    Simplified but captures the key dynamics:
    - Stochastic mid-price with regime switching
    - Order flow with adverse selection
    - Inventory risk
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self) -> MarketState:
        """Reset to start of new trading day."""
        self.mid = 100.0
        self.inventory = 0
        self.cash = 0.0
        self.step_count = 0
        self.max_steps = 2340
        self.regime = 'low'
        self.recent_buys = 0
        self.recent_sells = 0
        
        return self._get_state()
    
    def _get_state(self) -> MarketState:
        """Discretise current state."""
        # Inventory bins: <-200, -200 to -50, -50 to 50, 50 to 200, >200
        if self.inventory < -200:
            inv_bin = 0
        elif self.inventory < -50:
            inv_bin = 1
        elif self.inventory < 50:
            inv_bin = 2
        elif self.inventory < 200:
            inv_bin = 3
        else:
            inv_bin = 4
        
        vol_bin = 0 if self.regime == 'low' else 1
        
        progress = self.step_count / self.max_steps
        time_bin = 0 if progress < 0.33 else (1 if progress < 0.67 else 2)
        
        total_flow = self.recent_buys + self.recent_sells
        if total_flow > 0:
            imb = (self.recent_buys - self.recent_sells) / total_flow
        else:
            imb = 0
        imb_bin = 0 if imb < -0.2 else (2 if imb > 0.2 else 1)
        
        return MarketState(inv_bin, vol_bin, time_bin, imb_bin)
    
    def step(self, action: Action) -> Tuple[MarketState, float, bool]:
        """
        Execute one step.
        
        Returns:
            (next_state, reward, done)
        """
        self.step_count += 1
        
        # Regime switching
        if self.regime == 'low' and self.rng.random() < 0.005:
            self.regime = 'high'
        elif self.regime == 'high' and self.rng.random() < 0.02:
            self.regime = 'low'
        
        sigma = 0.012 if self.regime == 'low' else 0.04
        
        # Mid-price evolution
        self.mid += sigma * self.rng.standard_normal()
        self.mid = max(self.mid, 50)
        
        # Compute quotes from action
        half_spread = SPREAD_MAP[action.spread_level] * self.mid
        skew = SKEW_MAP[action.skew_level] * self.mid
        
        bid = self.mid - half_spread + skew
        ask = self.mid + half_spread + skew
        
        # Order flow
        prev_cash = self.cash
        prev_inv = self.inventory
        self.recent_buys = 0
        self.recent_sells = 0
        
        n_orders = self.rng.poisson(8)
        for _ in range(n_orders):
            side = 'buy' if self.rng.random() > 0.5 else 'sell'
            size = max(10, int(self.rng.normal(80, 20)))
            
            if side == 'buy':
                px = self.mid + self.rng.exponential(0.003) * self.mid
                if px >= ask:
                    self.cash += ask * size
                    self.inventory -= size
                    self.recent_sells += size
            else:
                px = self.mid - self.rng.exponential(0.003) * self.mid
                if px <= bid:
                    self.cash -= bid * size
                    self.inventory += size
                    self.recent_buys += size
        
        # Reward: change in MTM - inventory penalty
        mtm = self.cash + self.inventory * self.mid
        prev_mtm = prev_cash + prev_inv * self.mid
        pnl = mtm - prev_mtm
        
        # Inventory penalty (quadratic)
        inv_penalty = 0.0001 * self.inventory**2
        
        reward = pnl - inv_penalty
        
        done = self.step_count >= self.max_steps
        
        return self._get_state(), reward, done


# ============================================================================
# Q-Learning Agent
# ============================================================================

class QLearningAgent:
    """
    Tabular Q-learning agent for market-making.
    
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(
        self,
        n_states: int = MarketState.n_states(),
        n_actions: int = Action.n_actions(),
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.Q = np.zeros((n_states, n_actions))
        self.visit_counts = np.zeros((n_states, n_actions), dtype=int)
        self.rng = np.random.default_rng(42)
    
    def select_action(self, state: MarketState) -> Action:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            idx = self.rng.integers(self.n_actions)
        else:
            idx = np.argmax(self.Q[state.index])
        return Action.from_index(idx)
    
    def update(self, state: MarketState, action: Action,
               reward: float, next_state: MarketState, done: bool):
        """Q-learning update."""
        s = state.index
        a = action.index
        s_next = next_state.index
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[s_next])
        
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        self.visit_counts[s, a] += 1
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy(self) -> np.ndarray:
        """Get greedy policy (best action per state)."""
        return np.argmax(self.Q, axis=1)


# ============================================================================
# Training Loop
# ============================================================================

def train_agent(
    n_episodes: int = 500,
    steps_per_episode: int = 2340,
    seed: int = 42
) -> Tuple[QLearningAgent, dict]:
    """
    Train the RL market-maker.
    
    Returns:
        agent: trained Q-learning agent
        history: training metrics
    """
    agent = QLearningAgent()
    env = MMEnvironment(seed=seed)
    
    episode_rewards = []
    episode_pnls = []
    epsilons = []
    
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.decay_epsilon()
        
        final_pnl = env.cash + env.inventory * env.mid
        episode_rewards.append(total_reward)
        episode_pnls.append(final_pnl)
        epsilons.append(agent.epsilon)
        
        if (ep + 1) % 100 == 0:
            recent_pnl = np.mean(episode_pnls[-50:])
            print(f"Episode {ep+1:>4d} | Avg P&L: ${recent_pnl:>12,.0f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Avg reward: {np.mean(episode_rewards[-50:]):>10,.0f}")
    
    return agent, {
        'episode_rewards': np.array(episode_rewards),
        'episode_pnls': np.array(episode_pnls),
        'epsilons': np.array(epsilons),
    }


def evaluate_agent(agent: QLearningAgent, n_episodes: int = 100, seed: int = 999) -> dict:
    """Evaluate trained agent (no exploration)."""
    env = MMEnvironment(seed=seed)
    agent.epsilon = 0.0  # Pure exploitation
    
    pnls = []
    inventories = []
    
    for ep in range(n_episodes):
        env.rng = np.random.default_rng(seed + ep)
        state = env.reset()
        
        for step in range(2340):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            if done:
                break
        
        pnls.append(env.cash + env.inventory * env.mid)
        inventories.append(abs(env.inventory))
    
    pnls = np.array(pnls)
    return {
        'mean_pnl': np.mean(pnls),
        'std_pnl': np.std(pnls),
        'sharpe': np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0,
        'win_rate': np.mean(pnls > 0),
        'avg_final_inventory': np.mean(inventories),
        'pnl_distribution': pnls,
    }


def analyse_learned_policy(agent: QLearningAgent) -> dict:
    """Analyse what the agent learned."""
    policy = agent.get_policy()
    
    insights = {}
    
    # What spread does the agent use by inventory level?
    for inv_bin, inv_label in enumerate(['Very Short', 'Short', 'Flat', 'Long', 'Very Long']):
        actions = []
        for vol in range(2):
            for time_bin in range(3):
                for imb in range(3):
                    state = MarketState(inv_bin, vol, time_bin, imb)
                    a = Action.from_index(policy[state.index])
                    actions.append(a)
        
        avg_spread = np.mean([SPREAD_MAP[a.spread_level] for a in actions])
        avg_skew = np.mean([SKEW_MAP[a.skew_level] for a in actions])
        insights[inv_label] = {'avg_spread': avg_spread, 'avg_skew': avg_skew}
    
    # Spread by vol regime
    for vol, vol_label in [(0, 'Low Vol'), (1, 'High Vol')]:
        actions = []
        for inv in range(5):
            for time_bin in range(3):
                for imb in range(3):
                    state = MarketState(inv, vol, time_bin, imb)
                    a = Action.from_index(policy[state.index])
                    actions.append(a)
        avg_spread = np.mean([SPREAD_MAP[a.spread_level] for a in actions])
        insights[vol_label] = {'avg_spread': avg_spread}
    
    return insights


if __name__ == "__main__":
    print("Reinforcement Learning Market-Maker")
    print("=" * 50)
    
    t0 = time.time()
    agent, train_history = train_agent(n_episodes=500, steps_per_episode=2340)
    train_time = time.time() - t0
    print(f"\nTraining time: {train_time:.1f}s")
    
    # Evaluate
    print(f"\nEVALUATION (100 episodes, no exploration)")
    eval_results = evaluate_agent(agent, n_episodes=100)
    print(f"Mean P&L:        ${eval_results['mean_pnl']:,.0f}")
    print(f"Std P&L:         ${eval_results['std_pnl']:,.0f}")
    print(f"Sharpe:          {eval_results['sharpe']:.1f}")
    print(f"Win rate:        {eval_results['win_rate']:.0%}")
    print(f"Avg |inventory|: {eval_results['avg_final_inventory']:.0f}")
    
    # Policy analysis
    print(f"\nLEARNED POLICY ANALYSIS")
    print("-" * 40)
    insights = analyse_learned_policy(agent)
    
    for label, stats in insights.items():
        spread_bps = stats['avg_spread'] * 10000
        if 'avg_skew' in stats:
            skew_bps = stats['avg_skew'] * 10000
            print(f"{label:<15} spread: {spread_bps:>5.0f} bps, skew: {skew_bps:>+5.0f} bps")
        else:
            print(f"{label:<15} spread: {spread_bps:>5.0f} bps")

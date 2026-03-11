import gymnasium as gym
import numpy as np
import pickle
import json
from pathlib import Path
import os
from symmetries import RandomPermutation



class QLearningAgent:
    """An agent that learns a policy using Q-learning."""

    def __init__(self, action_space, rng, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.999):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("QLearningAgent requires a Discrete action space.")
        
        self.action_space = action_space
        self.num_actions = self.action_space.n
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay_rate
        self.q_table = {}
        self.rng = rng

    def get_state_q_values(self, state):
        """Safely get Q-values for a state, initializing if not present."""
        return self.q_table.setdefault(state, np.zeros(self.num_actions))

    def choose_action(self, state):
        """Choose an action using an epsilon-greedy policy."""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.num_actions)
        else:
            q_values = self.get_state_q_values(state)
            return np.argmax(q_values)

    def update(self, state, action, reward, next_state, terminated):
        """Update the Q-table using the Q-learning update rule."""
        q_values = self.get_state_q_values(state).copy()
        old_value = q_values[action]
        
        if terminated:
            next_max = 0
        else:
            next_max = np.max(self.get_state_q_values(next_state))
        
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        q_values[action] = new_value
        self.q_table[state] = q_values

    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save the Q-table to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        """Load a Q-table from a file."""
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)


def train(env, agent1, agent2, episodes):
    """Train two independent Q-learning agents in a shared environment."""
    for _ in range(episodes):
        state, _ = env.reset()
        state = tuple(state)
        
        action1 = agent1.choose_action(state)
        action2 = agent2.choose_action(state)
        
        next_state, reward, terminated, truncated, _ = env.step((action1, action2))
        next_state = tuple(next_state)
        done = terminated or truncated

        agent1.update(state, action1, reward, next_state, done)
        agent2.update(state, action2, reward, next_state, done)

        # Decay epsilon for both agents
        agent1.decay_epsilon()
        agent2.decay_epsilon()


def evaluate(env, agent1, agent2, n_rollouts=100):
    """Evaluate two agents' performance by averaging rewards over several rollouts."""
    total_reward = 0
    env.reset(seed=42)  # Use a fixed seed for comparable evaluations
    for _ in range(n_rollouts):
        state, _ = env.reset()
        state = tuple(state)
        
        # Use a greedy policy for evaluation
        action1 = np.argmax(agent1.get_state_q_values(state))
        action2 = np.argmax(agent2.get_state_q_values(state))
        
        _, reward, _, _, _ = env.step((action1, action2))
        total_reward += reward
        
    return total_reward / n_rollouts


def main():
    """
    Main function to train, evaluate, and save Q-learning agents for a two-player game.
    """
    # --- Hyperparameters ---
    N_SEEDS = 5
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.99
    # Epsilon decay parameters
    EXPLORATION_RATE = 1.0  # Start high
    MIN_EXPLORATION_RATE = 0.1
    EXPLORATION_DECAY_RATE = 0.9995 # decay per episode
    N_EPISODES = 5000
    N_EVAL_ROLLOUTS = 100
    
    # --- Setup ---
    script_dir = Path(__file__).parent
    trained_agents_dir = script_dir / "trained_agents"
    
    # --- Environment ---
    env = TwoPlayerLeverGame()
    
    if not isinstance(env.action_space, gym.spaces.Tuple) or len(env.action_space.spaces) != 2:
        raise ValueError("This script requires a 2-player environment with a Tuple action space.")

    trained_agents = []

    print(f"=== Training {N_SEEDS} seeds ===")
    for i in range(N_SEEDS):
        seed = i
        env.reset(seed=seed)  # Seed the environment for reproducibility
        rng = np.random.default_rng(seed)
        
        print(f"--- Training Seed {i+1}/{N_SEEDS} ---")
        
        agent1 = QLearningAgent(action_space=env.action_space.spaces[0],
                                rng=rng,
                                learning_rate=LEARNING_RATE,
                                discount_factor=DISCOUNT_FACTOR,
                                exploration_rate=EXPLORATION_RATE,
                                min_exploration_rate=MIN_EXPLORATION_RATE,
                                exploration_decay_rate=EXPLORATION_DECAY_RATE)
        
        agent2 = QLearningAgent(action_space=env.action_space.spaces[1],
                                rng=rng,
                                learning_rate=LEARNING_RATE,
                                discount_factor=DISCOUNT_FACTOR,
                                exploration_rate=EXPLORATION_RATE,
                                min_exploration_rate=MIN_EXPLORATION_RATE,
                                exploration_decay_rate=EXPLORATION_DECAY_RATE)
                                
        train(env, agent1, agent2, N_EPISODES)
        
        # Store agents in memory for cross-play evaluation
        trained_agents.append((agent1, agent2))
        
        # Save agents to disk
        seed_dir = trained_agents_dir / f"seed_{i}"
        agent1.save(seed_dir / "lever_game_q_learning_agent1.pkl")
        agent2.save(seed_dir / "lever_game_q_learning_agent2.pkl")

    print(f"Successfully trained {N_SEEDS} seeds")

    print()
    # --- Cross-Play Evaluation ---
    print("\n=== Cross-Play Evaluation ===")
    print(f"Evaluating {N_SEEDS} seeds against each other...")
    cross_play_matrix = np.zeros((N_SEEDS, N_SEEDS))
    
    for i in range(N_SEEDS):
        for j in range(N_SEEDS):
            # Agent 1 from seed i, Agent 2 from seed j
            agent1_from_seed_i = trained_agents[i][0]
            agent2_from_seed_j = trained_agents[j][1]
            
            avg_reward = evaluate(env, agent1_from_seed_i, agent2_from_seed_j, N_EVAL_ROLLOUTS)
            cross_play_matrix[i, j] = avg_reward

    # --- Calculate Final Metrics ---
    self_play_rewards = np.diag(cross_play_matrix)
    avg_self_play_reward = np.mean(self_play_rewards)
    
    mask = ~np.eye(N_SEEDS, dtype=bool)
    cross_play_rewards = cross_play_matrix[mask]
    avg_cross_play_reward = np.mean(cross_play_rewards) if len(cross_play_rewards) > 0 else 0
    print("Successfully evaluated cross-play")
    print()
    print(f"=== Cross-Play Results ===")
    print(f"Cross-play matrix: \n{cross_play_matrix}")
    
    print(f"\nAverage Self-Play Reward: {avg_self_play_reward:.2f}")
    print(f"Average Cross-Play Reward: {avg_cross_play_reward:.2f}")
    print(f"Self-play Cross-play gap: {avg_self_play_reward - avg_cross_play_reward:.2f}")
    
    # --- Save Results ---
    results = {
        "cross_play_matrix": cross_play_matrix.tolist(),
        "avg_self_play_reward": avg_self_play_reward,
        "avg_cross_play_reward": avg_cross_play_reward
    }
    
    results_file = script_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    env.close()


if __name__ == "__main__":
    from lever_game import TwoPlayerLeverGame
    main()

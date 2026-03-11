import numpy as np
from collections import defaultdict
import gymnasium as gym
import os
import pickle
import json

class IndependentQLearning:
    """
    Implements Independent Q-Learning for a 2-player fully cooperative turn-based game.
    
    This implementation assumes:
    - The game is for two players who play sequentially.
    - An episode consists of Player 0 taking one action, followed by Player 1 taking one action.
    - The environment provides a 'valid_actions' list in the info dictionary returned by `reset` and `step`.
    - Observations are convertible to tuples to be used as dictionary keys.
    """

    def __init__(self,
                 env: gym.Env,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995):
        """
        Initializes the Independent Q-Learning agent.

        Args:
            env: The gymnasium environment.
            learning_rate: The learning rate (alpha).
            discount_factor: The discount factor (gamma).
            epsilon_start: The starting value of epsilon for epsilon-greedy exploration.
            epsilon_end: The minimum value of epsilon.
            epsilon_decay: The decay rate of epsilon per episode.
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-tables for 2 players. Using defaultdict for convenience.
        # The key is the observation tuple, value is a numpy array of Q-values for each action.
        self.q_tables = [
            defaultdict(lambda: np.zeros(self.env.action_space.n)),
            defaultdict(lambda: np.zeros(self.env.action_space.n))
        ]

    def choose_action(self, agent_id: int, state: tuple, valid_actions: list) -> int:
        """
        Chooses an action for an agent using an epsilon-greedy policy.
        Handles random tie-breaking for the best action.
        """
        if not valid_actions:
            raise ValueError("No valid actions provided.")

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            q_values = self.q_tables[agent_id][state]
            # Create a dict of Q-values for valid actions only
            valid_q_values = {action: q_values[action] for action in valid_actions}
            max_q = max(valid_q_values.values())
            # Get all actions that have the max Q-value
            best_actions = [action for action, q in valid_q_values.items() if q == max_q]
            return np.random.choice(best_actions) # Randomly choose among the best actions

    def _update_epsilon(self):
        """Decays epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, num_episodes: int):
        """
        Trains the agents for a given number of episodes.
        """
        for _ in range(num_episodes):
            # --- Player 0's turn ---
            obs0, info0 = self.env.reset()
            state0 = tuple(obs0)
            valid_actions0 = info0.get('valid_actions')
            if valid_actions0 is None:
                raise KeyError("Info dict from env.reset() must contain 'valid_actions'.")

            action0 = self.choose_action(0, state0, valid_actions0)
            obs1, reward0, terminated, _, info1 = self.env.step(action0)
            
            if terminated:
                # Game ended after Player 0's move.
                # Update only Player 0's Q-table. Future reward is 0.
                q0 = self.q_tables[0]
                old_q0 = q0[state0][action0]
                update_target = reward0
                q0[state0][action0] = old_q0 + self.lr * (update_target - old_q0)
                self._update_epsilon()
                continue  # Go to the next episode

            # --- Player 1's turn ---
            state1 = tuple(obs1)
            valid_actions1 = info1.get('valid_actions')
            if valid_actions1 is None:
                raise KeyError("Info dict from env.step() must contain 'valid_actions'.")

            action1 = self.choose_action(1, state1, valid_actions1)
            _, reward1, terminated, _, _ = self.env.step(action1)
            
            # For fully cooperative games, both agents learn from the total discounted reward.
            total_discounted_reward = reward0 + self.gamma * reward1

            # --- Q-value updates (Monte Carlo style with shared reward) ---
            
            # Update Player 1's Q-table
            q1 = self.q_tables[1]
            old_q1 = q1[state1][action1]
            q1[state1][action1] = old_q1 + self.lr * (total_discounted_reward - old_q1)

            # Update Player 0's Q-table
            q0 = self.q_tables[0]
            old_q0 = q0[state0][action0]
            q0[state0][action0] = old_q0 + self.lr * (total_discounted_reward - old_q0)

            self._update_epsilon()

    def _greedy_action(self, q_table, state, valid_actions):
        if not valid_actions:
            raise ValueError("No valid actions provided for greedy selection.")
        
        q_values = q_table[state]
        valid_q_values = {action: q_values[action] for action in valid_actions}
        max_q = max(valid_q_values.values())
        best_actions = [action for action, q in valid_q_values.items() if q == max_q]
        return np.random.choice(best_actions)

    def evaluate(self, q_table0, q_table1, num_episodes: int, eval_seed: int):
        """
        Evaluate a pair of policies (Q-tables).
        
        Args:
            q_table0: Q-table for player 0.
            q_table1: Q-table for player 1.
            num_episodes: Number of episodes to run for evaluation.
            eval_seed: Seed for the environment to ensure consistent evaluation.

        Returns:
            Average total discounted reward over the episodes.
        """
        total_rewards = []
        
        # Seed the environment to make the evaluation reproducible
        obs0, info0 = self.env.reset(seed=eval_seed)

        for _ in range(num_episodes):
            # --- Player 0's turn ---
            state0 = tuple(obs0)
            valid_actions0 = info0.get('valid_actions')
            action0 = self._greedy_action(q_table0, state0, valid_actions0)
            
            obs1, reward0, terminated, _, info1 = self.env.step(action0)
            
            episode_reward = reward0
            
            if not terminated:
                # --- Player 1's turn ---
                state1 = tuple(obs1)
                valid_actions1 = info1.get('valid_actions')
                action1 = self._greedy_action(q_table1, state1, valid_actions1)
                
                _, reward1, _, _, _ = self.env.step(action1)
                episode_reward += self.gamma * reward1
            
            total_rewards.append(episode_reward)
            
            # Reset for the next episode in the evaluation run
            obs0, info0 = self.env.reset()

        return np.mean(total_rewards)


if __name__ == '__main__':
    from cat_dog import CatDogGame
    import copy

    # --- Hyperparameters ---
    n_seeds = 5
    num_training_episodes = 50000
    num_eval_episodes = 1000
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.9995

    # --- Setup ---
    env = CatDogGame()
    trained_policies = []  # List to store (q_table0, q_table1) from each seed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir_base = os.path.join(script_dir, 'trained_agents')

    # --- Training Phase ---
    print(f"--- Training {n_seeds} different policies ---")
    for i in range(n_seeds):
        seed = i
        print(f"\nTraining with seed {seed}...")

        # Set seed for reproducibility of training run
        env.reset(seed=seed)
        np.random.seed(seed)

        # Initialize a new Q-learner for each training run
        q_learner = IndependentQLearning(
            env=env,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay
        )

        q_learner.train(num_training_episodes)

        # Store a deepcopy of the trained Q-tables
        q_table_alice = copy.deepcopy(q_learner.q_tables[0])
        q_table_bob = copy.deepcopy(q_learner.q_tables[1])
        trained_policies.append((q_table_alice, q_table_bob))
        print(f"Finished training for seed {seed}.")

        # Save Q-tables
        save_dir = os.path.join(save_dir_base, f"seed_{seed}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "q_table_alice.pkl"), "wb") as f:
            pickle.dump(dict(q_table_alice), f)
        with open(os.path.join(save_dir, "q_table_bob.pkl"), "wb") as f:
            pickle.dump(dict(q_table_bob), f)

    print("\n--- Training complete ---")

    # --- Evaluation Phase (Cross-Play) ---
    print(f"\n--- Performing Cross-Play Evaluation ({n_seeds}x{n_seeds}) ---")
    cross_play_matrix = np.zeros((n_seeds, n_seeds))

    # Create one evaluator instance to use its env and methods
    evaluator = IndependentQLearning(env=env, discount_factor=discount_factor)

    for i in range(n_seeds):  # P0 from seed i
        for j in range(n_seeds):  # P1 from seed j
            q_table0 = trained_policies[i][0]
            q_table1 = trained_policies[j][1]

            # Use a consistent seed for all evaluations to compare fairly
            eval_seed = n_seeds
            avg_reward = evaluator.evaluate(q_table0, q_table1, num_eval_episodes, eval_seed=eval_seed)
            cross_play_matrix[i, j] = avg_reward

    # --- Results ---
    print("\n--- Cross-Play Matrix (Avg. Total Reward) ---")
    print("Rows: P0's policy seed, Columns: P1's policy seed")
    print(np.round(cross_play_matrix, 2))

    # --- Analysis and Saving Results ---
    # Calculate average self-play (diagonal) and cross-play (off-diagonal) rewards
    self_play_rewards = np.diag(cross_play_matrix)
    avg_self_play_reward = np.mean(self_play_rewards)

    # Create a mask for the off-diagonal elements
    off_diagonal_mask = ~np.eye(n_seeds, dtype=bool)
    cross_play_rewards = cross_play_matrix[off_diagonal_mask]
    # Handle case where n_seeds=1, so there are no off-diagonal elements
    avg_cross_play_reward = np.mean(cross_play_rewards) if cross_play_rewards.size > 0 else 0.0

    print(f"\nAverage Self-Play Reward (diagonal): {avg_self_play_reward:.2f}")
    print(f"Average Cross-Play Reward (off-diagonal): {avg_cross_play_reward:.2f}")
    print(f"Performance Gap (self-play - cross-play): {avg_self_play_reward - avg_cross_play_reward:.2f}")

    # Prepare results for JSON serialization
    results_data = {
        'cross_play_matrix': cross_play_matrix.tolist(),
        'avg_self_play_reward': avg_self_play_reward,
        'avg_cross_play_reward': avg_cross_play_reward
    }

    # Save results to a JSON file
    results_path = os.path.join(script_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=4)

    # print(f"\nResults summary saved to {results_path}")

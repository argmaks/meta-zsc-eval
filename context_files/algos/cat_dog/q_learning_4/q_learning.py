import numpy as np
import gymnasium as gym
from collections import defaultdict
import pickle
from typing import Dict, List, Tuple, Optional, Any
import copy


class TurnBasedTabularQLearning:
    """
    Independent tabular Q-learning for turn-based multi-agent games.
    Uses the observation to determine which agent is currently acting.
    """
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01,
        turn_index: int = 1,  # Index in observation that indicates whose turn it is
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize the Q-learning agent.
        
        Args:
            env: Gymnasium environment
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            turn_index: Index in observation array that indicates current player's turn
            rng: Numpy random generator for reproducible results
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.turn_index = turn_index
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Get environment specs
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
        # Q-tables for each agent (agent identified by turn value)
        self.q_tables: Dict[int, defaultdict] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_info = []
        
    def _get_state_key(self, observation: np.ndarray) -> str:
        """Convert observation to hashable state key."""
        # Convert to tuple for hashing
        return str(tuple(observation.astype(int)))
    
    def _get_current_agent(self, observation: np.ndarray) -> int:
        """Extract current agent from observation."""
        return int(observation[self.turn_index])
    
    def _get_valid_actions(self, info: Dict[str, Any]) -> List[int]:
        """Get valid actions from environment info, fallback to all actions."""
        if 'valid_actions' in info:
            return info['valid_actions']
        else:
            # Fallback: assume all actions in action space are valid
            if isinstance(self.action_space, gym.spaces.Discrete):
                return list(range(self.action_space.n))
            else:
                raise NotImplementedError("Only Discrete action spaces supported")
    
    def get_q_value(self, agent: int, state: str, action: int) -> float:
        """Get Q-value for given agent, state, and action."""
        return self.q_tables[agent][state][action]
    
    def update_q_value(self, agent: int, state: str, action: int, value: float):
        """Update Q-value for given agent, state, and action."""
        self.q_tables[agent][state][action] = value
    
    def choose_action(self, observation: np.ndarray, info: Dict[str, Any], training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            info: Environment info dict
            training: Whether in training mode (affects exploration)
        
        Returns:
            Selected action
        """
        agent = self._get_current_agent(observation)
        state_key = self._get_state_key(observation)
        valid_actions = self._get_valid_actions(info)
        
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Epsilon-greedy action selection
        if training and self.rng.random() < self.epsilon:
            # Explore: random action from valid actions
            return self.rng.choice(valid_actions)
        else:
            # Exploit: best action from valid actions
            q_values = [self.get_q_value(agent, state_key, action) for action in valid_actions]
            best_action_idx = np.argmax(q_values)
            return valid_actions[best_action_idx]
    
    def update(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool, Dict, Dict]):
        """
        Update Q-values based on experience.
        
        Args:
            experience: (state, action, reward, next_state, done, info, next_info)
        """
        state, action, reward, next_state, done, info, next_info = experience
        
        agent = self._get_current_agent(state)
        state_key = self._get_state_key(state)
        
        current_q = self.get_q_value(agent, state_key, action)
        
        if done:
            # Terminal state - use the provided reward (will be total episodic reward)
            target_q = reward
        else:
            # Get next agent and next state
            next_agent = self._get_current_agent(next_state)
            next_state_key = self._get_state_key(next_state)
            next_valid_actions = self._get_valid_actions(next_info)
            
            if next_valid_actions:
                # Maximum Q-value for next state
                next_q_values = [self.get_q_value(next_agent, next_state_key, a) for a in next_valid_actions]
                max_next_q = max(next_q_values)
            else:
                max_next_q = 0
            
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.update_q_value(agent, state_key, action, new_q)
    
    def update_cooperative(self, experiences: List[Tuple[np.ndarray, int, float, np.ndarray, bool, Dict, Dict]], total_reward: float):
        """
        Update Q-values for a fully cooperative episode using total episodic reward.
        Each agent acts only once per episode and is updated based on their own Q-values only.
        
        Args:
            experiences: List of (state, action, reward, next_state, done, info, next_info) tuples
            total_reward: Total reward for the entire episode
        """
        if not experiences:
            return
        
        # Since each agent acts only once per episode, we can directly update
        # each agent's Q-value using the total episodic reward
        for state, action, step_reward, next_state, done, info, next_info in experiences:
            agent = self._get_current_agent(state)
            state_key = self._get_state_key(state)
            
            current_q = self.get_q_value(agent, state_key, action)
            
            # Since each agent acts only once per episode, treat their action
            # as directly leading to the final outcome (total episodic reward)
            target_q = total_reward
            
            # Q-learning update - no need to consider next agent's Q-values
            # since each agent only acts once
            new_q = current_q + self.learning_rate * (target_q - current_q)
            self.update_q_value(agent, state_key, action, new_q)

    def train_episode(self) -> Tuple[float, int]:
        """
        Train for one episode using cooperative learning.
        
        Returns:
            Total reward and episode length
        """
        observation, info = self.env.reset()
        total_reward = 0
        episode_length = 0
        
        experiences = []
        
        while True:
            # Choose action
            action = self.choose_action(observation, info, training=True)
            
            # Take step
            next_observation, reward, terminated, truncated, next_info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            experience = (observation, action, reward, next_observation, done, info, next_info)
            experiences.append(experience)
            
            total_reward += reward
            episode_length += 1
            
            if done:
                break
                
            observation = next_observation
            info = next_info
        
        # Update Q-values for all experiences using total episodic reward (cooperative learning)
        self.update_cooperative(experiences, total_reward)
        
        return total_reward, episode_length
    
    def train(self, num_episodes: int, verbose: bool = True, eval_interval: int = 100) -> Dict[str, List]:
        """
        Train the agent for multiple episodes.
        
        Args:
            num_episodes: Number of training episodes
            verbose: Whether to print progress
            eval_interval: How often to evaluate performance
        
        Returns:
            Training statistics
        """
        for episode in range(num_episodes):
            total_reward, episode_length = self.train_episode()
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Evaluation and logging
            if episode % eval_interval == 0 and verbose:
                avg_reward = np.mean(self.episode_rewards[-eval_interval:])
                avg_length = np.mean(self.episode_lengths[-eval_interval:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.1f}, Epsilon: {self.epsilon:.3f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_info': self.training_info
        }
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation statistics
        """
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            observation, info = self.env.reset()
            total_reward = 0
            episode_length = 0
            
            while True:
                action = self.choose_action(observation, info, training=False)
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                total_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(total_reward)
            eval_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'std_length': np.std(eval_lengths),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards)
        }
    
    def get_policy(self, agent: int) -> Dict[str, int]:
        """
        Get the current policy for a specific agent.
        
        Args:
            agent: Agent identifier
        
        Returns:
            Dictionary mapping states to best actions
        """
        policy = {}
        
        for state in self.q_tables[agent]:
            q_values = self.q_tables[agent][state]
            if q_values:
                best_action = max(q_values.items(), key=lambda x: x[1])[0]
                policy[state] = best_action
        
        return policy
    
    def copy_agent_policy(self, agent: int) -> Dict[str, Dict[int, float]]:
        """
        Copy the Q-table for a specific agent.
        
        Args:
            agent: Agent identifier
        
        Returns:
            Deep copy of the agent's Q-table
        """
        agent_q_table = {}
        for state in self.q_tables[agent]:
            agent_q_table[state] = dict(self.q_tables[agent][state])
        return agent_q_table
    
    def set_agent_policy(self, agent: int, q_table: Dict[str, Dict[int, float]]):
        """
        Set the Q-table for a specific agent.
        
        Args:
            agent: Agent identifier
            q_table: Q-table to set for the agent
        """
        self.q_tables[agent] = defaultdict(lambda: defaultdict(float))
        for state, actions in q_table.items():
            for action, q_value in actions.items():
                self.q_tables[agent][state][action] = q_value
    
    def evaluate_with_policies(self, agent_policies: Dict[int, Dict[str, Dict[int, float]]], 
                              num_episodes: int = 100, eval_rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """
        Evaluate using specific agent policies.
        
        Args:
            agent_policies: Dictionary mapping agent_id to their Q-tables
            num_episodes: Number of evaluation episodes
            eval_rng: Random generator for evaluation (for reproducible results)
        
        Returns:
            Evaluation statistics
        """
        if eval_rng is None:
            eval_rng = np.random.default_rng()
        
        # Temporarily store current policies
        original_policies = {}
        for agent_id in agent_policies:
            original_policies[agent_id] = self.copy_agent_policy(agent_id)
        
        # Set the evaluation policies
        for agent_id, policy in agent_policies.items():
            self.set_agent_policy(agent_id, policy)
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            # Set environment seed for reproducible evaluation
            observation, info = self.env.reset(seed=int(eval_rng.integers(0, 2**31)))
            total_reward = 0
            episode_length = 0
            
            while True:
                action = self.choose_action(observation, info, training=False)
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                total_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(total_reward)
            eval_lengths.append(episode_length)
        
        # Restore original policies
        for agent_id, policy in original_policies.items():
            self.set_agent_policy(agent_id, policy)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'std_length': np.std(eval_lengths),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards)
        }
    
    def save(self, filepath: str):
        """Save the trained Q-tables and training history."""
        data = {
            'q_tables': dict(self.q_tables),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'turn_index': self.turn_index
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load trained Q-tables and training history."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Convert back to defaultdict format
        self.q_tables = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for agent, states in data['q_tables'].items():
            for state, actions in states.items():
                for action, q_value in actions.items():
                    self.q_tables[agent][state][action] = q_value
        
        self.episode_rewards = data['episode_rewards']
        self.episode_lengths = data['episode_lengths']
        
        # Update hyperparameters if available
        if 'hyperparameters' in data:
            params = data['hyperparameters']
            self.learning_rate = params['learning_rate']
            self.discount_factor = params['discount_factor']
            self.epsilon = params['epsilon']
            self.epsilon_decay = params['epsilon_decay']
            self.epsilon_min = params['epsilon_min']
            self.turn_index = params['turn_index']


def train_multiple_seeds(env_class, n_seeds: int, num_episodes: int = 10000, 
                        base_seed: int = 42, **agent_kwargs) -> List[Dict[int, Dict[str, Dict[int, float]]]]:
    """
    Train multiple policy pairs under different random seeds.
    
    Args:
        env_class: Environment class to instantiate
        n_seeds: Number of different seeds to train
        num_episodes: Number of training episodes per seed
        base_seed: Base seed for generating different seeds (not used, seeds go from 0 to n_seeds-1)
        **agent_kwargs: Additional arguments for the QLearning agent
    
    Returns:
        List of trained policies (one per seed), where each policy is a dict mapping agent_id to Q-table
    """
    import os
    
    trained_policies = []
    
    # Create base directory for trained agents
    base_dir = os.path.dirname(__file__)
    trained_agents_dir = os.path.join(base_dir, 'trained_agents')
    os.makedirs(trained_agents_dir, exist_ok=True)
    
    for seed in range(n_seeds):
        print(f"Training seed {seed+1}/{n_seeds} with seed {seed}")
        
        # Create environment and RNG for this seed
        env = env_class()
        training_rng = np.random.default_rng(seed)
        
        # Set environment seed
        env.reset(seed=seed)
        
        # Create agent with this seed's RNG
        agent = TurnBasedTabularQLearning(
            env=env,
            rng=training_rng,
            **agent_kwargs
        )
        
        # Train the agent
        agent.train(num_episodes=num_episodes, verbose=False)
        
        # Create directory for this seed
        seed_dir = os.path.join(trained_agents_dir, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        
        # Extract and save policies for both agents
        policies = {}
        agent_names = {0: 'alice', 1: 'bob'}
        
        for agent_id in [0, 1]:  # Assuming 2 agents (Alice=0, Bob=1)
            policies[agent_id] = agent.copy_agent_policy(agent_id)
            
            # Save individual agent Q-table
            agent_name = agent_names[agent_id]
            agent_file = os.path.join(seed_dir, f'{agent_name}.pkl')
            
            agent_data = {
                'q_table': policies[agent_id],
                'agent_id': agent_id,
                'agent_name': agent_name,
                'seed': seed,
                'training_params': {
                    'num_episodes': num_episodes,
                    'agent_kwargs': agent_kwargs
                }
            }
            
            with open(agent_file, 'wb') as f:
                pickle.dump(agent_data, f)
            
        
        trained_policies.append(policies)
        
        print(f"Completed training seed {seed+1}/{n_seeds}")
    
    return trained_policies


def evaluate_crossplay(env_class, trained_policies: List[Dict[int, Dict[str, Dict[int, float]]]], 
                      num_eval_episodes: int = 1000, eval_base_seed: int = 12345, 
                      **agent_kwargs) -> np.ndarray:
    """
    Evaluate cross-play between all pairs of trained policies.
    
    Args:
        env_class: Environment class to instantiate
        trained_policies: List of trained policies from different seeds
        num_eval_episodes: Number of episodes for each cross-play evaluation
        eval_base_seed: Base seed for evaluation RNG
        **agent_kwargs: Additional arguments for the QLearning agent
    
    Returns:
        n_seeds x n_seeds matrix where entry (i,j) is the average reward when
        agent 0 uses policy from seed i and agent 1 uses policy from seed j
    """
    n_seeds = len(trained_policies)
    crossplay_scores = np.zeros((n_seeds, n_seeds))
    
    # Create evaluation environment and RNG
    env = env_class()
    eval_rng = np.random.default_rng(eval_base_seed)
    
    # Create agent for evaluation (policies will be swapped in)
    agent = TurnBasedTabularQLearning(
        env=env,
        rng=eval_rng,
        **agent_kwargs
    )
    
    print(f"Evaluating {n_seeds}x{n_seeds} cross-play combinations...")
    
    for i in range(n_seeds):
        for j in range(n_seeds):

            
            # Create cross-play policy combination
            crossplay_policy = {
                0: trained_policies[i][0],  # Agent 0 from seed i
                1: trained_policies[j][1],  # Agent 1 from seed j
            }
            
            # Evaluate this combination
            results = agent.evaluate_with_policies(
                agent_policies=crossplay_policy,
                num_episodes=num_eval_episodes,
                eval_rng=eval_rng
            )
            
            crossplay_scores[i, j] = results['mean_reward']
    
    return crossplay_scores




def main():
    """Example usage with multi-seed training and cross-play evaluation."""
    import sys
    import os
    
    # Add the environment to the path
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'envs')
    sys.path.append(env_path)
    
    try:
        from cat_dog import CatDogGame
        
        # Training parameters
        n_seeds = 5
        num_episodes = 50000
        base_seed = 42
        
        # Agent parameters
        agent_params = {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.9995,
            'epsilon_min': 0.1,
            'turn_index': 1
        }
        
        print(f"Training {n_seeds} policy pairs...")
        trained_policies = train_multiple_seeds(
            env_class=CatDogGame,
            n_seeds=n_seeds,
            num_episodes=num_episodes,
            base_seed=base_seed,
            **agent_params
        )
        
        print(f"\nEvaluating cross-play...")
        crossplay_scores = evaluate_crossplay(
            env_class=CatDogGame,
            trained_policies=trained_policies,
            num_eval_episodes=1000,
            eval_base_seed=42,
            **agent_params
        )

        # Display cross-play evaluation results in LLM-friendly format
        print("\n" + "="*60)
        print("CROSS-PLAY EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nCROSS-PLAY MATRIX:")
        print(f"- Rows represent Agent 0 (Alice) policy seed")
        print(f"- Columns represent Agent 1 (Bob) policy seed") 
        print(f"- Values are average cooperative rewards")
        print("\nMatrix (Agent0_seed vs Agent1_seed):")
        for i in range(n_seeds):
            row_str = f"Seed_{i}: "
            for j in range(n_seeds):
                row_str += f"{crossplay_scores[i, j]:6.2f} "
            print(row_str)
        
        # Compute summary statistics
        self_play_scores = np.diag(crossplay_scores)
        off_diag_mask = ~np.eye(n_seeds, dtype=bool)
        cross_play_scores = crossplay_scores[off_diag_mask]
        
        avg_self_play_reward = float(np.mean(self_play_scores))
        avg_cross_play_reward = float(np.mean(cross_play_scores))
        gap = avg_self_play_reward - avg_cross_play_reward
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"- Self-play performance (same seed pairs): {avg_self_play_reward:.3f}")
        print(f"- Cross-play performance (different seed pairs): {avg_cross_play_reward:.3f}")
        print(f"- Performance gap (self-play - cross-play): {gap:.3f}")

        # Create results dictionary
        results_dict = {
            'cross_play_matrix': crossplay_scores.tolist(),
            'avg_self_play_reward': avg_self_play_reward,
            'avg_cross_play_reward': avg_cross_play_reward
        }
        
        # Save results to JSON file
        import json
        results_file = os.path.join(os.path.dirname(__file__), 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        
        
    except ImportError as e:
        print(f"Could not import CatDogGame: {e}")
        print("Make sure the environment is properly installed.")


if __name__ == "__main__":
    main()

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle
import os
import json
from symmetries import RandomPermutation


class TabularQLearningAgent:
    """
    Environment-agnostic tabular Q-learning agent that can work with any gym environment.
    Supports both single and multi-agent scenarios through independent learning.
    """
    
    def __init__(
        self,
        action_space: spaces.Space,
        observation_space: spaces.Space,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.1,
        agent_id: str = "agent_0"
    ):
        """
        Initialize the Q-learning agent.
        
        Args:
            action_space: The action space for this agent
            observation_space: The observation space
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial epsilon for epsilon-greedy exploration
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            agent_id: Unique identifier for this agent
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.agent_id = agent_id
        
        # Initialize Q-table
        self.q_table = {}
        
        # Statistics
        self.episode_rewards = []
        self.episode_count = 0
        
        # Validate action space
        if isinstance(action_space, spaces.Discrete):
            self.num_actions = action_space.n
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """
        Convert state array to a hashable key for the Q-table.
        
        Args:
            state: State observation
            
        Returns:
            String key for Q-table
        """
        # Handle different observation space types
        if isinstance(self.observation_space, spaces.Box):
            # For Box spaces, discretize continuous values
            if self.observation_space.dtype == np.float32:
                # Round to 3 decimal places for floating point states
                rounded_state = np.round(state, 3)
                return str(tuple(rounded_state.flatten()))
            else:
                return str(tuple(state.flatten()))
        elif isinstance(self.observation_space, spaces.Discrete):
            return str(int(state))
        else:
            # Fallback: convert to string
            return str(tuple(state.flatten()))
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions in the given state.
        
        Args:
            state: Current state
            
        Returns:
            Array of Q-values for all actions
        """
        state_key = self._state_to_key(state)
        if state_key not in self.q_table:
            # Initialize Q-values to zero for new states
            self.q_table[state_key] = np.zeros(self.num_actions)
        return self.q_table[state_key]
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        q_values = self.get_q_values(state)
        
        if training and np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.randint(self.num_actions)
        else:
            # Exploit: choose best action (break ties randomly)
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            return np.random.choice(max_actions)
    
    def update_q_value(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Update Q-value using the Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        state_key = self._state_to_key(state)
        
        # Get current Q-value
        current_q = self.get_q_values(state)[action]
        
        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            next_q_values = self.get_q_values(next_state)
            target_q = reward + self.discount_factor * np.max(next_q_values)
        
        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_episode(self):
        """Reset episode-specific variables."""
        self.episode_count += 1
    
    def save(self, filepath: str):
        """Save the agent's Q-table and parameters."""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_count': self.episode_count,
            'agent_id': self.agent_id,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load the agent's Q-table and parameters."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.episode_rewards = data['episode_rewards']
        self.episode_count = data['episode_count']
        self.agent_id = data['agent_id']


class IndependentQLearningTrainer:
    """
    Trainer for independent Q-learning agents in multi-agent environments.
    Environment-agnostic and works with any gym environment structure.
    """
    
    def __init__(
        self,
        env: gym.Env,
        num_agents: int = None,
        agent_configs: Dict[str, Any] = None
    ):
        """
        Initialize the multi-agent trainer.
        
        Args:
            env: Gym environment
            num_agents: Number of agents (auto-detected if None)
            agent_configs: Configuration for agents
        """
        self.env = env
        
        # Auto-detect number of agents from action space
        if num_agents is None:
            if isinstance(env.action_space, spaces.Tuple):
                self.num_agents = len(env.action_space.spaces)
            else:
                self.num_agents = 1
        else:
            self.num_agents = num_agents
        
        # Default agent configuration
        default_config = {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon': 1,
            'epsilon_decay': 0.9995,
            'epsilon_min': 0.1
        }
        
        if agent_configs is None:
            agent_configs = default_config
        else:
            # Merge with defaults
            for key, value in default_config.items():
                if key not in agent_configs:
                    agent_configs[key] = value
        
        # Create agents
        self.agents = []
        for i in range(self.num_agents):
            if isinstance(env.action_space, spaces.Tuple):
                agent_action_space = env.action_space.spaces[i]
            else:
                agent_action_space = env.action_space
            
            agent = TabularQLearningAgent(
                action_space=agent_action_space,
                observation_space=env.observation_space,
                agent_id=f"agent_{i}",
                **agent_configs
            )
            self.agents.append(agent)
        
        # Training statistics
        self.training_rewards = []
        self.training_episodes = 0
    
    def train(
        self,
        num_episodes: int,
        max_steps_per_episode: int = 1000,
        verbose: bool = True,
        log_interval: int = 100
    ) -> Dict[str, List[float]]:
        """
        Train the agents using independent Q-learning.
        
        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            verbose: Whether to print progress
            log_interval: Interval for logging progress
            
        Returns:
            Dictionary with training statistics
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Reset environment
            observation, info = self.env.reset()
            episode_reward = 0
            
            # Reset agents
            for agent in self.agents:
                agent.reset_episode()
            
            for step in range(max_steps_per_episode):
                # Each agent selects an action independently
                actions = []
                for agent in self.agents:
                    action = agent.select_action(observation, training=True)
                    actions.append(action)
                
                # Convert to environment action format
                if self.num_agents == 1:
                    env_action = actions[0]
                else:
                    env_action = tuple(actions)
                
                # Take environment step
                next_observation, reward, terminated, truncated, info = self.env.step(env_action)
                done = terminated or truncated
                
                # Update each agent's Q-table
                for i, agent in enumerate(self.agents):
                    agent.update_q_value(observation, actions[i], reward, next_observation, done)
                
                observation = next_observation
                episode_reward += reward
                
                if done:
                    break
            
            # Decay epsilon for all agents
            for agent in self.agents:
                agent.decay_epsilon()
            
            episode_rewards.append(episode_reward)
            self.training_rewards.append(episode_reward)
            self.training_episodes += 1
            
        if verbose:
            print(f"Finished training!")
        
        return {
            'episode_rewards': episode_rewards,
            'final_epsilon': [agent.epsilon for agent in self.agents]
        }
    
    def evaluate(
        self,
        num_episodes: int = 100,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate the trained agents.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render the environment
            
        Returns:
            Evaluation statistics
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                # Each agent selects action greedily (no exploration)
                actions = []
                for agent in self.agents:
                    action = agent.select_action(observation, training=False)
                    actions.append(action)
                
                # Convert to environment action format
                if self.num_agents == 1:
                    env_action = actions[0]
                else:
                    env_action = tuple(actions)
                
                observation, reward, terminated, truncated, info = self.env.step(env_action)
                episode_reward += reward
                steps += 1
                
                if render:
                    self.env.render()
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'episode_rewards': episode_rewards
        }
    
    def evaluate_with_specific_agents(
        self,
        agents: List[TabularQLearningAgent],
        num_episodes: int = 100,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate using specific agents instead of the trainer's own agents.
        
        Args:
            agents: List of agents to use for evaluation
            num_episodes: Number of evaluation episodes
            render: Whether to render the environment
            
        Returns:
            Evaluation statistics
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                # Each agent selects action greedily (no exploration)
                actions = []
                for agent in agents:
                    action = agent.select_action(observation, training=False)
                    actions.append(action)
                
                # Convert to environment action format
                if len(agents) == 1:
                    env_action = actions[0]
                else:
                    env_action = tuple(actions)
                
                observation, reward, terminated, truncated, info = self.env.step(env_action)
                episode_reward += reward
                steps += 1
                
                if render:
                    self.env.render()
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'episode_rewards': episode_rewards
        }
    
    def save_agents(self, save_dir: str):
        """Save all agents with the expected naming convention."""
        os.makedirs(save_dir, exist_ok=True)
        for i, agent in enumerate(self.agents):
            # Use the naming convention expected by evaluate.py
            filepath = os.path.join(save_dir, f"lever_game_q_learning_agent{i+1}.pkl")
            agent.save(filepath)
    
    def load_agents(self, save_dir: str):
        """Load all agents with the expected naming convention."""
        for i, agent in enumerate(self.agents):
            # Try the expected naming convention first
            filepath = os.path.join(save_dir, f"lever_game_q_learning_agent{i+1}.pkl")
            if os.path.exists(filepath):
                agent.load(filepath)
            else:
                # Fallback to old naming convention
                filepath = os.path.join(save_dir, f"agent_{i}.pkl")
                if os.path.exists(filepath):
                    agent.load(filepath)


class CrossPlayAnalyzer:
    """
    Analyzer for cross-play evaluation of independently trained agents.
    Trains multiple agent pairs with different seeds and evaluates all cross-play combinations.
    """
    
    def __init__(
        self,
        env_factory,
        agent_configs: Dict[str, Any] = None,
        save_dir: str = None
    ):
        """
        Initialize the cross-play analyzer.
        
        Args:
            env_factory: Function that creates a new environment instance
            agent_configs: Configuration for agents
            save_dir: Directory to save trained agents (defaults to script directory)
        """
        self.env_factory = env_factory
        self.agent_configs = agent_configs or {}
        
        # Default to trained_agents in the same directory as the script
        if save_dir is None:
            script_dir = os.path.dirname(__file__)
            self.save_dir = os.path.join(script_dir, "trained_agents")
        else:
            self.save_dir = save_dir
            
        self.trained_agents = {}  # {seed: [agent1, agent2, ...]}
        self.training_stats = {}  # {seed: training_stats}
        
    def train_multiple_seeds(
        self,
        seeds: List[int],
        num_episodes: int,
        verbose: bool = True,
        save_agents: bool = True
    ) -> Dict[int, Dict]:
        """
        Train agent pairs for multiple random seeds.
        
        Args:
            seeds: List of random seeds to use
            num_episodes: Number of training episodes per seed
            verbose: Whether to print progress
            save_agents: Whether to save trained agents
            
        Returns:
            Dictionary with training statistics for each seed
        """
        all_stats = {}
        
        for seed in seeds:
            if verbose:
                print(f"\n=== Training agents with seed {seed} ===")
            
            # Create environment with seed
            env = self.env_factory()
            np.random.seed(seed)
            
            # Create trainer
            trainer = IndependentQLearningTrainer(env, agent_configs=self.agent_configs)
            
            # Train agents
            train_stats = trainer.train(
                num_episodes=num_episodes,
                verbose=verbose,
                log_interval=max(1, num_episodes // 10)
            )
            
            # Store agents and stats
            self.trained_agents[seed] = trainer.agents.copy()
            self.training_stats[seed] = train_stats
            all_stats[seed] = train_stats
            
            # Save agents if requested
            if save_agents:
                seed_dir = os.path.join(self.save_dir, f"seed_{seed}")
                trainer.save_agents(seed_dir)
        
        return all_stats
    
    def evaluate_cross_play(
        self,
        seeds: List[int] = None,
        num_episodes: int = 100,
        verbose: bool = True,
        save_results: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate all cross-play combinations of trained agents.
        
        Args:
            seeds: List of seeds to include (None = all trained seeds)
            num_episodes: Number of evaluation episodes per combination
            verbose: Whether to print progress
            save_results: Whether to save results to results.json
            
        Returns:
            Dictionary containing cross-play matrices and statistics
        """
        if seeds is None:
            seeds = list(self.trained_agents.keys())
        
        n_seeds = len(seeds)
        
        # Initialize result matrices
        reward_matrix = np.zeros((n_seeds, n_seeds))
        std_matrix = np.zeros((n_seeds, n_seeds))
        length_matrix = np.zeros((n_seeds, n_seeds))
        
        # Create a fresh environment for evaluation
        eval_env = self.env_factory()
        trainer = IndependentQLearningTrainer(eval_env)
        
        if verbose:
            print(f"\n=== Cross-play evaluation for {n_seeds} seeds ===")
            print(f"Evaluating {n_seeds * n_seeds} cross-play combinations")
        
        for i, seed_i in enumerate(seeds):
            for j, seed_j in enumerate(seeds):
                
                # Get agents from different seeds
                agent1 = self.trained_agents[seed_i][0]  # First agent from seed_i
                
                if len(self.trained_agents[seed_j]) > 1:
                    agent2 = self.trained_agents[seed_j][1]  # Second agent from seed_j
                else:
                    agent2 = self.trained_agents[seed_j][0]  # Single agent case
                
                # Evaluate this combination
                eval_stats = trainer.evaluate_with_specific_agents(
                    agents=[agent1, agent2],
                    num_episodes=num_episodes,
                    render=False
                )
                
                # Store results
                reward_matrix[i, j] = eval_stats['mean_reward']
                std_matrix[i, j] = eval_stats['std_reward']
                length_matrix[i, j] = eval_stats['mean_length']
                
        if verbose:
            print(f"Finished evaluating!")
        
        results = {
            'reward_matrix': reward_matrix,
            'std_matrix': std_matrix,
            'length_matrix': length_matrix,
            'seeds': seeds,
            'self_play_rewards': np.diag(reward_matrix),
            'cross_play_rewards': reward_matrix[np.triu_indices(n_seeds, k=1)],
            'mean_self_play': np.mean(np.diag(reward_matrix)),
            'mean_cross_play': np.mean(reward_matrix[np.triu_indices(n_seeds, k=1)]),
            'generalization_gap': np.mean(np.diag(reward_matrix)) - np.mean(reward_matrix[np.triu_indices(n_seeds, k=1)])
        }
        
        # Save results to JSON file if requested
        if save_results:
            self.save_results_json(results)
        
        return results
    
    def save_results_json(self, results: Dict[str, Any]):
        """
        Save cross-play results to results.json in the expected format.
        
        Args:
            results: Results dictionary from evaluate_cross_play
        """
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'cross_play_matrix': results['reward_matrix'].tolist(),
            'avg_self_play_reward': float(results['mean_self_play']),
            'avg_cross_play_reward': float(results['mean_cross_play']),
            'generalization_gap': float(results['generalization_gap']),
        }
        
        # Save to the same directory as the save_dir
        results_file = os.path.join(os.path.dirname(__file__), "results.json")
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # print(f"Results saved to {results_file}")
    
    def load_agents_from_seeds(self, seeds: List[int]):
        """
        Load previously saved agents from disk.
        
        Args:
            seeds: List of seeds to load
        """
        for seed in seeds:
            seed_dir = os.path.join(self.save_dir, f"seed_{seed}")
            if os.path.exists(seed_dir):
                # Create dummy trainer to load agents
                env = self.env_factory()
                trainer = IndependentQLearningTrainer(env, agent_configs=self.agent_configs)
                trainer.load_agents(seed_dir)
                self.trained_agents[seed] = trainer.agents.copy()
            else:
                print(f"Warning: No saved agents found for seed {seed} in {seed_dir}")
    
    def print_cross_play_summary(self, results: Dict[str, np.ndarray]):
        """
        Print a formatted summary of cross-play results.
        
        Args:
            results: Results from evaluate_cross_play
        """
        reward_matrix = results['reward_matrix']
        seeds = results['seeds']
        n_seeds = len(seeds)
        
        print(f"\n=== Cross-Play Results Summary ===")
        print(f"Self-play performance: {results['mean_self_play']:.3f}")
        print(f"Cross-play performance: {results['mean_cross_play']:.3f}")
        print(f"Self-play - Cross-play gap: {results['generalization_gap']:.3f}")
        
        print(f"\nReward Matrix (rows=Agent1 seed, cols=Agent2 seed):")
        print("Seeds:", [f"S{s}" for s in seeds])
        
        # Print header
        print("     ", end="")
        for seed in seeds:
            print(f"  S{seed:2d}", end="")
        print()
        
        # Print matrix rows
        for i, seed_i in enumerate(seeds):
            print(f"S{seed_i:2d}: ", end="")
            for j in range(n_seeds):
                print(f"{reward_matrix[i, j]:5.2f}", end="")
            print()


def create_trainer_for_env(env: gym.Env, **kwargs) -> IndependentQLearningTrainer:
    """
    Factory function to create a trainer for any gym environment.
    
    Args:
        env: Gym environment
        **kwargs: Additional arguments for the trainer
        
    Returns:
        Configured trainer
    """
    return IndependentQLearningTrainer(env, **kwargs)


def create_cross_play_analyzer(env_factory, **kwargs) -> CrossPlayAnalyzer:
    """
    Factory function to create a cross-play analyzer.
    
    Args:
        env_factory: Function that creates environment instances
        **kwargs: Additional arguments for the analyzer
        
    Returns:
        Configured cross-play analyzer
    """
    return CrossPlayAnalyzer(env_factory, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Import the lever game environment
    from lever_game import TwoPlayerLeverGame
    
    print("Training Independent Q-Learning Agents")
    print("=" * 40)
    
    # Define environment factory
    def env_factory():
        return TwoPlayerLeverGame(random_seed=None)
    
    # Create cross-play analyzer
    analyzer = create_cross_play_analyzer(
        env_factory,
        agent_configs={
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.9995,
            'epsilon_min': 0.01
        }
    )
    
    # Train agents with multiple seeds
    seeds = [0, 1, 2, 3, 4]
    analyzer.train_multiple_seeds(
        seeds=seeds,
        num_episodes=5000,
        verbose=True
    )
    
    # Evaluate cross-play
    cross_play_results = analyzer.evaluate_cross_play(
        seeds=seeds,
        num_episodes=100,
        verbose=True
    )
    
    # Print summary
    analyzer.print_cross_play_summary(cross_play_results)

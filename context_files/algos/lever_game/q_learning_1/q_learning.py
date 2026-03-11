import numpy as np
from collections import defaultdict
import sys
import os
import pickle
import json

class IndependentQLearningAgent:
    """
    Independent Q-learning agent for multi-agent environments.
    Each agent learns independently without considering other agents' actions.
    """
    
    def __init__(self, action_space_size, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.1):
        """
        Initialize the Q-learning agent.
        
        Args:
            action_space_size: Number of possible actions
            learning_rate: Learning rate (alpha) for Q-learning updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate for epsilon-greedy policy
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action -> Q-value
        # Using defaultdict to handle unseen states
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        
        # Statistics tracking
        self.total_reward = 0
        self.episode_count = 0
        
    def state_to_key(self, state):
        """Convert state observation to a hashable key for Q-table."""
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        return tuple([state]) if not isinstance(state, tuple) else state
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        state_key = self.state_to_key(state)
        
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.action_space_size)
        else:
            # Exploit: action with highest Q-value
            return np.argmax(self.q_table[state_key])
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Q-learning update rule
        current_q = self.q_table[state_key][action]
        
        if done:
            # Terminal state: no future rewards
            max_next_q = 0
        else:
            # Maximum Q-value for next state
            max_next_q = np.max(self.q_table[next_state_key])
        
        # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        updated_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = updated_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_action_probabilities(self, state):
        """Get action probabilities for the current policy."""
        state_key = self.state_to_key(state)
        q_values = self.q_table[state_key]
        
        # Softmax for visualization
        exp_q = np.exp(q_values - np.max(q_values))
        return exp_q / np.sum(exp_q)
    
    def save_q_table(self, filepath):
        """
        Save the Q-table and agent parameters to a file.
        
        Args:
            filepath: Path to save the Q-table (relative to script location if not absolute)
        """
        # Make path relative to script location if not absolute
        if not os.path.isabs(filepath):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, filepath)
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        agent_data = {
            'q_table': dict(self.q_table),  # Convert defaultdict to regular dict
            'action_space_size': self.action_space_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'total_reward': self.total_reward,
            'episode_count': self.episode_count
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        # print(f"Q-table saved to {filepath}")
    
    def load_q_table(self, filepath):
        """
        Load the Q-table and agent parameters from a file.
        
        Args:
            filepath: Path to load the Q-table from (relative to script location if not absolute)
        """
        # Make path relative to script location if not absolute
        if not os.path.isabs(filepath):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, filepath)
        
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        # Restore Q-table as defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))
        for state_key, q_values in agent_data['q_table'].items():
            self.q_table[state_key] = q_values
        
        # Restore other parameters
        self.action_space_size = agent_data['action_space_size']
        self.learning_rate = agent_data['learning_rate']
        self.discount_factor = agent_data['discount_factor']
        self.epsilon = agent_data['epsilon']
        self.epsilon_decay = agent_data['epsilon_decay']
        self.epsilon_min = agent_data['epsilon_min']
        self.total_reward = agent_data['total_reward']
        self.episode_count = agent_data['episode_count']
        
        print(f"Q-table loaded from {filepath}")


class IndependentQLearningTrainer:
    """
    Trainer for independent Q-learning in multi-agent environments.
    """
    
    def __init__(self, env, agent1_params=None, agent2_params=None, random_seed=None):
        """
        Initialize the trainer.
        
        Args:
            env: Multi-agent environment
            agent1_params: Parameters for agent 1
            agent2_params: Parameters for agent 2
            random_seed: Random seed for reproducibility
        """
        self.env = env
        
        # Set random seed for reproducibility
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Default parameters
        default_params = {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 1.0,
            'epsilon_decay': 0.9995,
            'epsilon_min': 0.1
        }
        
        agent1_params = agent1_params or default_params
        agent2_params = agent2_params or default_params
        
        # Create agents
        action_space_size = env.action_space[0].n  # Assuming both agents have same action space
        self.agent1 = IndependentQLearningAgent(action_space_size, **agent1_params)
        self.agent2 = IndependentQLearningAgent(action_space_size, **agent2_params)
        
        # Training statistics
        self.episode_rewards = []
        
    def train(self, num_episodes=1000, verbose=True, log_interval=100):
        """
        Train both agents using independent Q-learning.
        
        Args:
            num_episodes: Number of training episodes
            verbose: Whether to print training progress
            log_interval: How often to print progress
        """
        for episode in range(num_episodes):
            # Reset environment
            state, info = self.env.reset()
            
            # Get actions from both agents
            action1 = self.agent1.select_action(state, training=True)
            action2 = self.agent2.select_action(state, training=True)
            
            # Execute joint action
            joint_action = (action1, action2)
            next_state, reward, terminated, truncated, info = self.env.step(joint_action)
            done = terminated or truncated
            
            # Update both agents (they both receive the same reward in this environment)
            self.agent1.update_q_value(state, action1, reward, next_state, done)
            self.agent2.update_q_value(state, action2, reward, next_state, done)
            
            # Decay exploration rates
            self.agent1.decay_epsilon()
            self.agent2.decay_epsilon()
            
            # Track statistics
            self.episode_rewards.append(reward)
            
            # Update agent statistics
            self.agent1.total_reward += reward
            self.agent2.total_reward += reward
            self.agent1.episode_count += 1
            self.agent2.episode_count += 1
            
            # Logging
            if verbose and (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.3f}")
                print("-" * 40)
    
    def evaluate(self, num_episodes=100, verbose=True):
        """
        Evaluate the trained agents.
        
        Args:
            num_episodes: Number of evaluation episodes
            verbose: Whether to print evaluation results
            
        Returns:
            Dictionary with evaluation metrics
        """
        eval_rewards = []
        action_counts = {i: 0 for i in range(self.env.num_levers)}
        joint_action_counts = {}
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            
            # Get actions (no exploration)
            action1 = self.agent1.select_action(state, training=False)
            action2 = self.agent2.select_action(state, training=False)
            
            # Track action frequencies
            action_counts[int(action1)] += 1
            joint_key = (int(action1), int(action2))
            joint_action_counts[joint_key] = joint_action_counts.get(joint_key, 0) + 1
            
            # Execute and record
            joint_action = (action1, action2)
            next_state, reward, terminated, truncated, info = self.env.step(joint_action)
            
            eval_rewards.append(reward)
        
        # Calculate metrics
        metrics = {
            'avg_reward': np.mean(eval_rewards),
            'action_frequencies': {k: v/num_episodes for k, v in action_counts.items()},
            'joint_action_frequencies': {k: v/num_episodes for k, v in joint_action_counts.items()}
        }
        
        if verbose:
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(f"Average Reward: {metrics['avg_reward']:.3f}")
            print(f"\nLever Rewards: {self.env.rewards}")
            
            print(f"\nAction Frequencies:")
            for action, freq in metrics['action_frequencies'].items():
                print(f"  Lever {action}: {freq:.3f}")
            
            print(f"\nTop Joint Actions:")
            sorted_joint = sorted(metrics['joint_action_frequencies'].items(), 
                                key=lambda x: x[1], reverse=True)
            for (a1, a2), freq in sorted_joint[:5]:
                reward = self.env.rewards[a1] if a1 == a2 else 0
                print(f"  ({a1}, {a2}): {freq:.3f} (reward: {reward})")
        
        return metrics
    
    def save_training(self, filepath):
        """
        Save both agents' Q-tables and training statistics to a file.
        
        Args:
            filepath: Path to save the training data (without extension, relative to script location if not absolute)
        """
        # Make path relative to script location if not absolute
        if not os.path.isabs(filepath):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, filepath)
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save agent Q-tables
        self.agent1.save_q_table(f"{filepath}_agent1.pkl")
        self.agent2.save_q_table(f"{filepath}_agent2.pkl")
        
        
        # print(f"Training data saved with prefix: {filepath}")
    
    def load_training(self, filepath):
        """
        Load both agents' Q-tables and training statistics from files.
        
        Args:
            filepath: Path prefix to load the training data from (without extension, relative to script location if not absolute)
        """
        # Make path relative to script location if not absolute
        if not os.path.isabs(filepath):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, filepath)
        
        # Load agent Q-tables
        self.agent1.load_q_table(f"{filepath}_agent1.pkl")
        self.agent2.load_q_table(f"{filepath}_agent2.pkl")
        
        # Load training statistics
        try:
            with open(f"{filepath}_training_stats.pkl", 'rb') as f:
                training_data = pickle.load(f)
            
            self.episode_rewards = training_data['episode_rewards']
            
            print(f"Training statistics loaded from {filepath}_training_stats.pkl")
        except FileNotFoundError:
            print(f"Training statistics file not found: {filepath}_training_stats.pkl")
            print("Continuing with empty training statistics.")
        
        print(f"Training data loaded with prefix: {filepath}")
    


def main():
    """Main function to train multiple pairs of Q-learning agents and evaluate cross-play performance."""
    import sys
    import os
    from lever_game import TwoPlayerLeverGame

    
    print("Independent Tabular Q-Learning for Two-Player Lever Game")
    print("="*70)
    
    # Configuration
    n_seeds = 5  # Number of different seeds to train
    num_train_episodes = 5000
    num_eval_episodes = 1000
    
    print(f"Training {n_seeds} pairs of agents across different seeds...")
    print(f"Training episodes per pair: {num_train_episodes}")
    print(f"Evaluation episodes per combination: {num_eval_episodes}")
    print()
    
    # Create environment to get basic info
    env = TwoPlayerLeverGame()
    print(f"Environment: {env.__class__.__name__}")
    print(f"Number of levers: {env.num_levers}")
    print(f"Lever rewards: {env.rewards}")
    print()
    
    # Store all trained agents
    trained_agents = {}  # {seed: (agent1, agent2)}
    
    # Train agents for each seed
    for seed in range(n_seeds):
        print(f"Training agents for seed {seed}...")
        
        # Create environment with specific seed
        env = TwoPlayerLeverGame(random_seed=seed)
        
        # Create trainer with specific seed
        trainer = IndependentQLearningTrainer(env, random_seed=seed)
        
        # Train agents
        trainer.train(num_episodes=num_train_episodes, verbose=False)
        
        # Store the trained agents
        trained_agents[seed] = (trainer.agent1, trainer.agent2)
        
        # Save trained agents
        trainer.save_training(f"trained_agents/seed_{seed}/lever_game_q_learning")
        
        print(f"Completed training for seed {seed}")
    
    print("\nTraining completed for all seeds!")
    print("\nStarting cross-play evaluation...")
    
    # Cross-play evaluation
    cross_play_matrix = np.zeros((n_seeds, n_seeds))
    evaluation_details = {}
    
    # Evaluate all combinations of (agent1_seed_i, agent2_seed_j)
    for i in range(n_seeds):
        for j in range(n_seeds):
            # print(f"Evaluating Agent1(seed={i}) vs Agent2(seed={j})...")
            
            # Create fresh environment for evaluation
            eval_env = TwoPlayerLeverGame()
            
            # Get the agents
            agent1 = trained_agents[i][0]  # Agent1 from seed i
            agent2 = trained_agents[j][1]  # Agent2 from seed j
            
            # Evaluate this combination
            eval_rewards = []
            
            for episode in range(num_eval_episodes):
                state, info = eval_env.reset()
                
                # Get actions (no exploration)
                action1 = agent1.select_action(state, training=False)
                action2 = agent2.select_action(state, training=False)
                
                # Execute and record
                joint_action = (action1, action2)
                next_state, reward, terminated, truncated, info = eval_env.step(joint_action)
                
                eval_rewards.append(reward)
            
            # Store results
            avg_reward = np.mean(eval_rewards)
            
            cross_play_matrix[i, j] = avg_reward
            evaluation_details[(i, j)] = {
                'avg_reward': avg_reward,
                'agent1_seed': i,
                'agent2_seed': j
            }
    
    # Calculate self-play and cross-play averages
    self_play_rewards = np.diag(cross_play_matrix)  # Diagonal elements
    avg_self_play_reward = np.mean(self_play_rewards)
    
    # Cross-play rewards (off-diagonal elements)
    cross_play_rewards = cross_play_matrix[~np.eye(n_seeds, dtype=bool)]
    avg_cross_play_reward = np.mean(cross_play_rewards)
    
    # Print results
    print("\n" + "="*70)
    print("CROSS-PLAY EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nCross-Play Matrix (Agent1_seed × Agent2_seed):")
    print("```")
    # Header row
    header = "Agent1\\Agent2"
    for j in range(n_seeds):
        header += f"    Seed{j}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for i in range(n_seeds):
        row = f"Seed{i}      "
        for j in range(n_seeds):
            if i == j:  # Self-play (diagonal)
                row += f"  {cross_play_matrix[i, j]:.3f}*"
            else:       # Cross-play (off-diagonal)
                row += f"  {cross_play_matrix[i, j]:.3f} "
        print(row)
    print("```")
    print("* = Self-play (same seed for both agents)")

    
    print(f"\nSummary Statistics:")
    print(f"Average Self-Play Reward: {avg_self_play_reward:.4f}")
    print(f"Average Cross-Play Reward: {avg_cross_play_reward:.4f}")
    print(f"Self-Play vs Cross-Play Difference: {avg_self_play_reward - avg_cross_play_reward:.4f}")
    
    # DO NOT MODIFY THE CODE BELOW THIS LINE
    
    # Return results
    results = {
        'cross_play_matrix': cross_play_matrix.tolist(),  
        'avg_self_play_reward': float(avg_self_play_reward),  
        'avg_cross_play_reward': float(avg_cross_play_reward),
    }

    # Save results to a json file in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_filepath = os.path.join(script_dir, 'results.json')
    
    if os.path.exists(results_filepath):
        try:
            with open(results_filepath, 'r') as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or empty, start fresh
            all_results = {}
    else:
        all_results = {}
    
    # Add current results to the dictionary
    all_results = all_results | results
    
    # Save updated results back to file
    with open(results_filepath, 'w') as f:
        json.dump(all_results, f, indent=2)

    # print(f"Results saved to {results_filepath}")

    return results


if __name__ == "__main__":
    results = main()

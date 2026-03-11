import numpy as np
import pickle
import os
import json
from collections import defaultdict
from cat_dog import CatDogGame


class IndependentQLearning:
    """
    Independent Q-Learning for the cooperative Cat Dog game.
    Each agent (Alice and Bob) learns their own Q-table independently.
    """
    
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1, epsilon_min=0.1, epsilon_decay=0.995, seed=None):
        """
        Initialize the Q-learning agents.
        
        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay factor per episode
            seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.initial_epsilon = epsilon
        self.seed = seed
        
        # Create isolated random number generator for this agent
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        
        # Q-tables for each agent using full observations
        self.q_alice = defaultdict(lambda: np.zeros(4))  # Alice has 4 actions
        self.q_bob = defaultdict(lambda: np.zeros(3))    # Bob has 3 actions
        
        # Training statistics
        self.episode_rewards = []
        self.alice_action_counts = defaultdict(int)
        self.bob_action_counts = defaultdict(int)
        self.epsilon_history = []
        
    def choose_action(self, observation, q_table, valid_actions, is_training=True):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            q_table: Q-table to use
            valid_actions: List of valid actions
            is_training: Whether in training mode (for exploration)
        """
        if is_training and self.rng.random() < self.epsilon:
            # Explore: choose random valid action using agent's RNG
            return self.rng.choice(valid_actions)
        else:
            # Exploit: choose action with highest Q-value among valid actions
            q_values = q_table[observation]
            # Mask invalid actions with very negative values
            masked_q_values = np.full_like(q_values, -np.inf)
            for action in valid_actions:
                masked_q_values[action] = q_values[action]
            return np.argmax(masked_q_values)
    
    def update_q_table(self, q_table, observation, action, reward, next_observation, next_valid_actions, done):
        """Update Q-table using Q-learning update rule."""
        current_q = q_table[observation][action]
        
        if done:
            # Terminal state
            max_next_q = 0
        else:
            # Get maximum Q-value for next state among valid actions
            next_q_values = q_table[next_observation]
            if next_valid_actions:
                max_next_q = max(next_q_values[a] for a in next_valid_actions)
            else:
                max_next_q = 0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        q_table[observation][action] = new_q
    
    def train(self, num_episodes=10000, verbose=True):
        """Train the Q-learning agents."""
        env = CatDogGame()
        
        for episode in range(num_episodes):
            # Seed the environment for each episode if we have a seed
            if self.seed is not None:
                # Use episode-specific seed for determinism while allowing variation
                episode_seed = self.seed * 100000 + episode
                observation, info = env.reset(seed=episode_seed)
            else:
                observation, info = env.reset()
            episode_reward = 0
            
            # Track observations and actions for updates
            alice_obs = None
            alice_action = None
            bob_obs = None
            bob_action = None
            
            while True:
                current_turn = observation[1]  # 0=Alice, 1=Bob
                valid_actions = info['valid_actions']
                
                if current_turn == 0:  # Alice's turn
                    alice_obs = tuple(observation)
                    alice_action = self.choose_action(alice_obs, self.q_alice, valid_actions)
                    self.alice_action_counts[alice_action] += 1
                    
                    next_observation, reward, terminated, truncated, next_info = env.step(alice_action)
                    episode_reward += reward
                    
                    if terminated:
                        # Game ended on Alice's action (she bailed out)
                        self.update_q_table(self.q_alice, alice_obs, alice_action, reward, 
                                          None, [], True)
                        break
                    else:
                        # Continue to Bob's turn - we'll update Alice's Q-table after Bob acts
                        observation = next_observation
                        info = next_info
                
                else:  # Bob's turn
                    bob_obs = tuple(observation)
                    bob_action = self.choose_action(bob_obs, self.q_bob, valid_actions)
                    self.bob_action_counts[bob_action] += 1
                    
                    next_observation, reward, terminated, truncated, next_info = env.step(bob_action)
                    episode_reward += reward
                    
                    # Update both Q-tables since game ends after Bob's action
                    # Alice gets the same reward (cooperative game)
                    if alice_obs is not None and alice_action is not None:
                        self.update_q_table(self.q_alice, alice_obs, alice_action, episode_reward, 
                                          None, [], True)
                    
                    # Update Bob's Q-table with total episode reward (cooperative)
                    self.update_q_table(self.q_bob, bob_obs, bob_action, episode_reward, 
                                      None, [], True)
                    break
            
            self.episode_rewards.append(episode_reward)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.epsilon_history.append(self.epsilon)
            
            # Print progress
            if verbose and (episode + 1) % 1000 == 0:
                avg_reward = np.mean(self.episode_rewards[-1000:])
                print(f"Episode {episode + 1}: Average reward (last 1000): {avg_reward:.3f}, Epsilon: {self.epsilon:.3f}")
    
    def evaluate(self, num_episodes=1000):
        """Evaluate the learned policies."""
        env = CatDogGame()
        eval_rewards = []
        action_combinations = defaultdict(int)
        
        for episode in range(num_episodes):
            # Use deterministic seed for evaluation if agent has a seed
            if self.seed is not None:
                eval_seed = self.seed * 50000 + episode  # Different offset from training
                observation, info = env.reset(seed=eval_seed)
            else:
                observation, info = env.reset()
            episode_reward = 0
            alice_action_taken = None
            bob_action_taken = None
            
            while True:
                current_turn = observation[1]
                valid_actions = info['valid_actions']
                
                if current_turn == 0:  # Alice's turn
                    alice_obs = tuple(observation)
                    alice_action = self.choose_action(alice_obs, self.q_alice, valid_actions, is_training=False)
                    alice_action_taken = alice_action
                    
                    next_observation, reward, terminated, truncated, next_info = env.step(alice_action)
                    episode_reward += reward
                    
                    if terminated:
                        break
                    else:
                        observation = next_observation
                        info = next_info
                
                else:  # Bob's turn
                    bob_obs = tuple(observation)
                    bob_action = self.choose_action(bob_obs, self.q_bob, valid_actions, is_training=False)
                    bob_action_taken = bob_action
                    
                    next_observation, reward, terminated, truncated, next_info = env.step(bob_action)
                    episode_reward += reward
                    break
            
            eval_rewards.append(episode_reward)
            if alice_action_taken is not None and bob_action_taken is not None:
                action_combinations[(alice_action_taken, bob_action_taken)] += 1
        
        return eval_rewards, action_combinations
    



def save_agent_policies(agent, save_dir, seed, prefix="agent", verbose=True):
    """
    Save an agent's Q-tables and training metadata to pickle files.
    
    Args:
        agent: Trained IndependentQLearning agent
        save_dir: Base directory to save in
        seed: Seed number for this agent
        prefix: Prefix for filenames
        verbose: Whether to print save confirmation
    """
    # Create directory structure
    agent_dir = os.path.join(save_dir, f"seed_{seed}")
    os.makedirs(agent_dir, exist_ok=True)
    
    # Save Alice's Q-table
    alice_path = os.path.join(agent_dir, f"{prefix}_alice.pkl")
    with open(alice_path, 'wb') as f:
        pickle.dump(dict(agent.q_alice), f)
    
    # Save Bob's Q-table  
    bob_path = os.path.join(agent_dir, f"{prefix}_bob.pkl")
    with open(bob_path, 'wb') as f:
        pickle.dump(dict(agent.q_bob), f)
    
    # Save training metadata
    metadata = {
        'seed': seed,
        'alpha': agent.alpha,
        'gamma': agent.gamma,
        'initial_epsilon': agent.initial_epsilon,
        'final_epsilon': agent.epsilon,
        'epsilon_min': agent.epsilon_min,
        'epsilon_decay': agent.epsilon_decay,
        'episode_rewards': agent.episode_rewards,
        'alice_action_counts': dict(agent.alice_action_counts),
        'bob_action_counts': dict(agent.bob_action_counts),
        'epsilon_history': agent.epsilon_history,
        'total_episodes': len(agent.episode_rewards)
    }
    
    metadata_path = os.path.join(agent_dir, f"{prefix}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    if verbose:
        print(f"Saved agent policies to {agent_dir}/")
    return agent_dir


def save_all_agents(agents, base_dir="trained_agents", prefix="agent"):
    """
    Save all trained agents to organized directory structure.
    
    Args:
        agents: List of trained IndependentQLearning agents
        base_dir: Base directory name
        prefix: Prefix for filenames
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, base_dir)
    
    saved_paths = []
    for i, agent in enumerate(agents):
        agent_dir = save_agent_policies(agent, save_dir, seed=i, prefix=prefix, verbose=False)
        saved_paths.append(agent_dir)
    
    
    return save_dir


def load_agent_policies(agent_dir, prefix="agent"):
    """
    Load an agent's Q-tables and metadata from pickle files.
    
    Args:
        agent_dir: Directory containing the agent files
        prefix: Prefix used when saving
        
    Returns:
        Tuple of (alice_q_table, bob_q_table, metadata)
    """
    # Load Alice's Q-table
    alice_path = os.path.join(agent_dir, f"{prefix}_alice.pkl")
    with open(alice_path, 'rb') as f:
        alice_q = pickle.load(f)
    
    # Load Bob's Q-table
    bob_path = os.path.join(agent_dir, f"{prefix}_bob.pkl")
    with open(bob_path, 'rb') as f:
        bob_q = pickle.load(f)
    
    # Load metadata
    metadata_path = os.path.join(agent_dir, f"{prefix}_metadata.pkl")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return alice_q, bob_q, metadata


def load_all_agents(base_dir="trained_agents", prefix="agent"):
    """
    Load all saved agents from directory structure.
    
    Args:
        base_dir: Base directory name
        prefix: Prefix used when saving
        
    Returns:
        List of (alice_q_table, bob_q_table, metadata) tuples
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dir = os.path.join(script_dir, base_dir)
    
    # Load summary to get seed information
    summary_path = os.path.join(load_dir, "training_summary.pkl")
    with open(summary_path, 'rb') as f:
        summary = pickle.load(f)
    
    agents_data = []
    for seed in summary['seeds']:
        agent_dir = os.path.join(load_dir, f"seed_{seed}")
        alice_q, bob_q, metadata = load_agent_policies(agent_dir, prefix)
        agents_data.append((alice_q, bob_q, metadata))
    
    return agents_data, summary


def create_agent_from_saved(alice_q, bob_q, metadata):
    """
    Recreate an IndependentQLearning agent from saved Q-tables and metadata.
    
    Args:
        alice_q: Alice's Q-table (dict)
        bob_q: Bob's Q-table (dict)
        metadata: Training metadata (dict)
        
    Returns:
        IndependentQLearning agent with loaded policies
    """
    agent = IndependentQLearning(
        alpha=metadata['alpha'],
        gamma=metadata['gamma'],
        epsilon=metadata['final_epsilon'],
        epsilon_min=metadata['epsilon_min'],
        epsilon_decay=metadata['epsilon_decay'],
        seed=metadata['seed']
    )
    
    # Load Q-tables
    agent.q_alice = defaultdict(lambda: np.zeros(4))
    agent.q_alice.update(alice_q)
    
    agent.q_bob = defaultdict(lambda: np.zeros(3))
    agent.q_bob.update(bob_q)
    
    # Load training history
    agent.episode_rewards = metadata['episode_rewards']
    agent.alice_action_counts = defaultdict(int)
    agent.alice_action_counts.update(metadata['alice_action_counts'])
    agent.bob_action_counts = defaultdict(int)
    agent.bob_action_counts.update(metadata['bob_action_counts'])
    agent.epsilon_history = metadata['epsilon_history']
    
    return agent


def train_multiple_agents(n_seeds, num_episodes=100000, alpha=0.1, gamma=0.95, 
                         epsilon=1, epsilon_min=0.1, epsilon_decay=0.995, verbose=True,
                         save_policies=True, save_dir="trained_agents"):
    """
    Train multiple agent pairs with different random seeds.
    
    Args:
        n_seeds: Number of different seeds to train
        num_episodes: Number of episodes to train each agent pair
        Other args: Q-learning hyperparameters
        verbose: Whether to print training progress
        save_policies: Whether to save trained policies to files
        save_dir: Directory name to save policies in
        
    Returns:
        List of trained agents
    """
    agents = []
    
    for seed in range(n_seeds):
        if verbose:
            print(f"Training agent pair {seed + 1}/{n_seeds} (seed={seed})...")
        
        agent = IndependentQLearning(
            alpha=alpha, gamma=gamma, epsilon=epsilon,
            epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, seed=seed
        )
        agent.train(num_episodes=num_episodes, verbose=False)  # Reduce verbosity for individual training
        agents.append(agent)
        
        if verbose:
            final_reward = np.mean(agent.episode_rewards[-1000:]) if len(agent.episode_rewards) >= 1000 else np.mean(agent.episode_rewards)
            print(f"Seed {seed} completed: Final reward={final_reward:.2f}, Epsilon={agent.epsilon:.3f}")
    
    # Save all trained agents if requested
    if save_policies:
        save_all_agents(agents, base_dir=save_dir, prefix="agent")
    
    return agents


def evaluate_cross_play(agents, num_episodes=1000, verbose=True):
    """
    Evaluate agents in cross-play: Alice from seed i paired with Bob from seed j.
    
    Args:
        agents: List of trained agents
        num_episodes: Number of episodes to evaluate each pair
        verbose: Whether to print progress
        
    Returns:
        Cross-play reward matrix (n_seeds x n_seeds)
    """
    n_seeds = len(agents)
    cross_play_matrix = np.zeros((n_seeds, n_seeds))
    
    if verbose:
        print(f"Evaluating {n_seeds}x{n_seeds} = {n_seeds**2} agent combinations...")
    
    for i in range(n_seeds):
        for j in range(n_seeds):
            # Create environment
            env = CatDogGame()
            episode_rewards = []
            
            for episode in range(num_episodes):
                # Use deterministic seed for evaluation based on agent pair and episode
                eval_seed = (i * 1000 + j) * 10000 + episode
                observation, info = env.reset(seed=eval_seed)
                episode_reward = 0
                
                while True:
                    current_turn = observation[1]
                    valid_actions = info['valid_actions']
                    
                    if current_turn == 0:  # Alice's turn
                        alice_obs = tuple(observation)
                        # Use Alice from agent i
                        alice_action = agents[i].choose_action(
                            alice_obs, agents[i].q_alice, valid_actions, is_training=False
                        )
                        
                        next_observation, reward, terminated, truncated, next_info = env.step(alice_action)
                        episode_reward += reward
                        
                        if terminated:
                            break
                        else:
                            observation = next_observation
                            info = next_info
                    
                    else:  # Bob's turn
                        bob_obs = tuple(observation)
                        # Use Bob from agent j
                        bob_action = agents[j].choose_action(
                            bob_obs, agents[j].q_bob, valid_actions, is_training=False
                        )
                        
                        next_observation, reward, terminated, truncated, next_info = env.step(bob_action)
                        episode_reward += reward
                        break
                
                episode_rewards.append(episode_reward)
            
            avg_reward = np.mean(episode_rewards)
            cross_play_matrix[i, j] = avg_reward
            
            if verbose:
                print(f"Alice(S{i}) + Bob(S{j}): {avg_reward:.3f}")
    
    return cross_play_matrix


def main():
    """Main training and evaluation function."""
    # Configuration
    n_seeds = 5  # Number of different agents to train
    num_episodes = 50000  # Number of episodes to train each agent
    eval_episodes = 1000  # Episodes for cross-play evaluation
    save_dir = "trained_agents"  # Directory to save trained policies
    
    print("INDEPENDENT Q-LEARNING: Cat Dog Game Cross-Play Evaluation")
    print(f"Training {n_seeds} agent pairs, {num_episodes} episodes each")
    
    # Train multiple agent pairs with different seeds
    agents = train_multiple_agents(
        n_seeds=n_seeds,
        num_episodes=num_episodes,
        alpha=0.1,
        gamma=0.95,
        epsilon=1,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        verbose=True,
        save_policies=True,
        save_dir=save_dir
    )
    
    
    # Cross-play evaluation
    print(f"\nCROSS-PLAY EVALUATION: {n_seeds}x{n_seeds} combinations, {eval_episodes} episodes each")
    cross_play_matrix = evaluate_cross_play(agents, num_episodes=eval_episodes, verbose=False)
    
    print("\nCross-play Results (Alice_seed vs Bob_seed):")
    print(f"\nMatrix Summary:")
    header = "Alice\\Bob  " + "  ".join(f"B{j}" for j in range(n_seeds))
    print(header)
    for i in range(n_seeds):
        row_values = "  ".join(f"{cross_play_matrix[i,j]:4.1f}" for j in range(n_seeds))
        print(f"A{i}      {row_values}")
    
    # Key statistics
    diagonal_mean = np.mean(np.diag(cross_play_matrix))
    off_diagonal_mean = np.mean(cross_play_matrix[~np.eye(n_seeds, dtype=bool)])
    
    print(f"\nAverage self-play reward={diagonal_mean:.2f}")
    print(f"Average cross-play reward={off_diagonal_mean:.2f}")
    print(f"Self-play cross-play difference={diagonal_mean - off_diagonal_mean:.2f}")

    # Save results to JSON file for evaluation script
    results = {
        "cross_play_matrix": cross_play_matrix.tolist(),  # Convert numpy array to list for JSON
        "avg_self_play_reward": float(diagonal_mean),
        "avg_cross_play_reward": float(off_diagonal_mean)
    }
    
    # Save to results.json in the evaluate/q_learning directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(script_dir, exist_ok=True)
    results_file = os.path.join(script_dir, "results.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    
    print("Evaluation complete.")
    

if __name__ == "__main__":
    main()
    

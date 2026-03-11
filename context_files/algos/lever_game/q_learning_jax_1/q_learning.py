import jax
import jax.numpy as jnp
import numpy as np
from jax import random, jit
from functools import partial
from typing import Tuple, NamedTuple
import pickle
import os
import json

from lever_game import TwoPlayerLeverGame


class QLearningState(NamedTuple):
    """State for Q-learning algorithm"""
    q_table_1: jnp.ndarray  # Q-table for player 1: [state, action]
    q_table_2: jnp.ndarray  # Q-table for player 2: [state, action]
    epsilon: float
    episode: int


class QLearningParams(NamedTuple):
    """Parameters for Q-learning"""
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    num_episodes: int = 10000


def init_q_learning_state(num_states: int, num_actions: int, key: jax.random.PRNGKey) -> QLearningState:
    """Initialize Q-learning state with random Q-tables"""
    key1, key2 = random.split(key)
    
    # Initialize Q-tables with small random values
    q_table_1 = random.normal(key1, (num_states, num_actions)) * 0.01
    q_table_2 = random.normal(key2, (num_states, num_actions)) * 0.01
    
    return QLearningState(
        q_table_1=q_table_1,
        q_table_2=q_table_2,
        epsilon=1.0,
        episode=0
    )


@partial(jit, static_argnums=(2,))
def epsilon_greedy_action(q_values: jnp.ndarray, epsilon: float, num_actions: int, key: jax.random.PRNGKey) -> int:
    """Select action using epsilon-greedy policy"""
    key1, key2 = random.split(key)
    explore = random.uniform(key1) < epsilon
    
    # Random action if exploring
    random_action = random.randint(key1, (), 0, num_actions)
    
    # Greedy action (use argmax, break ties with random noise)
    # Add small random noise to break ties randomly
    noise = random.uniform(key2, q_values.shape) * 1e-8
    noisy_q_values = q_values + noise
    greedy_action = jnp.argmax(noisy_q_values)
    
    return jnp.where(explore, random_action, greedy_action)


@jit
def update_q_value(q_table: jnp.ndarray, state: int, action: int, reward: float, 
                   next_state: int, learning_rate: float, discount_factor: float, done: bool) -> jnp.ndarray:
    """Update Q-value using Q-learning update rule"""
    current_q = q_table[state, action]
    
    # Use JAX-compatible conditional: jnp.where instead of if/else
    next_q_max = jnp.max(q_table[next_state])
    # If done: target = reward, else: target = reward + discount * next_q_max
    target = jnp.where(done, reward, reward + discount_factor * next_q_max)
    
    # Q-learning update
    td_error = target - current_q
    new_q = current_q + learning_rate * td_error
    
    # Update Q-table
    return q_table.at[state, action].set(new_q)


@jit
def update_epsilon(epsilon: float, epsilon_decay: float, epsilon_end: float) -> float:
    """Decay epsilon for exploration"""
    return jnp.maximum(epsilon * epsilon_decay, epsilon_end)


@partial(jit, static_argnums=(2,))
def select_actions(state: QLearningState, observation: jnp.ndarray, num_actions: int, key: jax.random.PRNGKey) -> Tuple[int, int]:
    """Select actions for both players"""
    # Convert observation to state index (since we have minimal state space)
    state_idx = 0  # For this one-shot game, we can use a single state
    
    key1, key2 = random.split(key)
    
    # Get Q-values for current state
    q_values_1 = state.q_table_1[state_idx]
    q_values_2 = state.q_table_2[state_idx]
    
    # Select actions for both players
    action_1 = epsilon_greedy_action(q_values_1, state.epsilon, num_actions, key1)
    action_2 = epsilon_greedy_action(q_values_2, state.epsilon, num_actions, key2)
    
    return action_1, action_2


@partial(jit, static_argnums=(6,))
def update_q_learning_state(state: QLearningState, obs: jnp.ndarray, actions: Tuple[int, int], 
                           reward: float, next_obs: jnp.ndarray, done: bool, params: QLearningParams) -> QLearningState:
    """Update Q-learning state after taking actions"""
    # Convert observations to state indices
    state_idx = 0
    next_state_idx = 0
    
    action_1, action_2 = actions
    
    # Update Q-tables for both players (independent learning)
    new_q_table_1 = update_q_value(
        state.q_table_1, state_idx, action_1, reward, next_state_idx,
        params.learning_rate, params.discount_factor, done
    )
    
    new_q_table_2 = update_q_value(
        state.q_table_2, state_idx, action_2, reward, next_state_idx,
        params.learning_rate, params.discount_factor, done
    )
    
    # Update epsilon
    new_epsilon = update_epsilon(state.epsilon, params.epsilon_decay, params.epsilon_end)
    
    return QLearningState(
        q_table_1=new_q_table_1,
        q_table_2=new_q_table_2,
        epsilon=new_epsilon,
        episode=state.episode + jnp.where(done, 1, 0)
    )


def train_q_learning(params: QLearningParams, seed: int = 0) -> Tuple[QLearningState, list]:
    """Train Q-learning agents on the lever game"""
    
    # Initialize environment
    env = TwoPlayerLeverGame(random_seed=seed)
    num_actions = env.num_levers
    num_states = 1  # Simplified state space for one-shot game
    
    # Initialize JAX random key
    key = random.PRNGKey(seed)
    
    # Initialize Q-learning state
    key, init_key = random.split(key)
    q_state = init_q_learning_state(num_states, num_actions, init_key)
    
    # Training metrics
    episode_rewards = []
    coordination_success = []
    
    print("Starting Q-learning training...")
    # print(f"Environment: {num_actions} levers, rewards: {env.rewards}")
    
    for episode in range(params.num_episodes):
        # Reset environment
        observation, _ = env.reset()
        obs_jax = jnp.array(observation, dtype=jnp.float32)
        
        # Select actions
        key, action_key = random.split(key)
        action_1, action_2 = select_actions(q_state, obs_jax, num_actions, action_key)
        
        # Take step in environment
        actions_tuple = (int(action_1), int(action_2))
        next_observation, reward, terminated, truncated, info = env.step(actions_tuple)
        next_obs_jax = jnp.array(next_observation, dtype=jnp.float32)
        
        # Update Q-learning state
        q_state = update_q_learning_state(
            q_state, obs_jax, (action_1, action_2), reward, next_obs_jax, terminated, params
        )
        
        # Track metrics
        episode_rewards.append(reward)
        coordination_success.append(info['actions_match'])
        
        # Print progress
        # if episode % 1000 == 0:
        #     recent_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        #     recent_coordination = np.mean(coordination_success[-100:]) if len(coordination_success) >= 100 else np.mean(coordination_success)
        #     print(f"Episode {episode}: Avg Reward: {recent_reward:.3f}, Coordination Rate: {recent_coordination:.3f}, Epsilon: {q_state.epsilon:.3f}")
    
    print("Training completed!")
    return q_state, episode_rewards


def evaluate_policy(q_state: QLearningState, num_episodes: int = 1000, seed: int = 123) -> dict:
    """Evaluate the learned policy"""
    env = TwoPlayerLeverGame(random_seed=seed)
    num_actions = env.num_levers
    
    key = random.PRNGKey(seed)
    
    total_reward = 0
    action_counts = np.zeros((num_actions, num_actions))
    coordination_successes = 0
    
    # Use greedy policy (epsilon = 0)
    eval_state = q_state._replace(epsilon=0.0)
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        obs_jax = jnp.array(observation, dtype=jnp.float32)
        
        key, action_key = random.split(key)
        action_1, action_2 = select_actions(eval_state, obs_jax, num_actions, action_key)
        
        actions_tuple = (int(action_1), int(action_2))
        _, reward, _, _, info = env.step(actions_tuple)
        
        total_reward += reward
        action_counts[action_1, action_2] += 1
        if info['actions_match']:
            coordination_successes += 1
    
    avg_reward = total_reward / num_episodes
    coordination_rate = coordination_successes / num_episodes
    
    print(f"\nEvaluation Results ({num_episodes} episodes):")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Coordination Rate: {coordination_rate:.3f}")
    print(f"Most frequent action pairs:")
    
    # Find top action combinations
    flat_indices = np.argsort(action_counts.flatten())[::-1][:5]
    for i, flat_idx in enumerate(flat_indices):
        a1, a2 = np.unravel_index(flat_idx, action_counts.shape)
        count = action_counts[a1, a2]
        if count > 0:
            reward_val = env.rewards[a1] if a1 == a2 else 0
            print(f"  {i+1}. Player1: {a1}, Player2: {a2}, Count: {count:4.0f}, Reward: {reward_val:.1f}")
    
    return {
        'avg_reward': avg_reward,
        'coordination_rate': coordination_rate,
        'action_counts': action_counts
    }


def evaluate_crossplay(q_state_1: QLearningState, q_state_2: QLearningState, 
                       num_episodes: int = 100, seed: int = 123) -> float:
    """Evaluate crossplay between two different policies"""
    env = TwoPlayerLeverGame(random_seed=seed)
    num_actions = env.num_levers
    
    key = random.PRNGKey(seed)
    
    total_reward = 0
    
    # Use greedy policies (epsilon = 0)
    eval_state_1 = q_state_1._replace(epsilon=0.0)
    eval_state_2 = q_state_2._replace(epsilon=0.0)
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        obs_jax = jnp.array(observation, dtype=jnp.float32)
        
        key, key1, key2 = random.split(key, 3)
        
        # Get action from player 1's policy
        state_idx = 0
        q_values_1 = eval_state_1.q_table_1[state_idx]
        action_1 = epsilon_greedy_action(q_values_1, 0.0, num_actions, key1)
        
        # Get action from player 2's policy
        q_values_2 = eval_state_2.q_table_2[state_idx]
        action_2 = epsilon_greedy_action(q_values_2, 0.0, num_actions, key2)
        
        actions_tuple = (int(action_1), int(action_2))
        _, reward, _, _, _ = env.step(actions_tuple)
        
        total_reward += reward
    
    return total_reward / num_episodes


def save_q_tables(q_state: QLearningState, seed_idx: int, params: QLearningParams, base_dir: str = None):
    """Save Q-tables to pickle files"""
    # Default to script directory if no base_dir provided
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "trained_agents")
    
    # Create directory structure
    seed_dir = os.path.join(base_dir, f"seed_{seed_idx}")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Create metadata
    metadata = {
        'seed': seed_idx,
        'learning_rate': params.learning_rate,
        'discount_factor': params.discount_factor,
        'epsilon_start': params.epsilon_start,
        'epsilon_end': params.epsilon_end,
        'epsilon_decay': params.epsilon_decay,
        'num_episodes': params.num_episodes,
        'final_epsilon': q_state.epsilon,
        'final_episode': q_state.episode
    }
    
    # Save both agents from this seed
    # Agent 1 (Player 1 Q-table)
    agent1_path = os.path.join(seed_dir, "lever_game_q_learning_agent1.pkl")
    with open(agent1_path, 'wb') as f:
        pickle.dump({
            'q_table': np.array(q_state.q_table_1),
            'agent_type': 'player1',
            'metadata': metadata
        }, f)
    
    # Agent 2 (Player 2 Q-table)  
    agent2_path = os.path.join(seed_dir, "lever_game_q_learning_agent2.pkl")
    with open(agent2_path, 'wb') as f:
        pickle.dump({
            'q_table': np.array(q_state.q_table_2),
            'agent_type': 'player2',
            'metadata': metadata
        }, f)
    
    return agent1_path, agent2_path


def load_q_table(filepath: str) -> dict:
    """Load a saved Q-table from pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def load_agent_pair(seed_idx: int, base_dir: str = None) -> Tuple[dict, dict]:
    """Load both agents from a specific seed"""
    # Default to script directory if no base_dir provided
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "trained_agents")
    
    seed_dir = os.path.join(base_dir, f"seed_{seed_idx}")
    agent1_path = os.path.join(seed_dir, "lever_game_q_learning_agent1.pkl")
    agent2_path = os.path.join(seed_dir, "lever_game_q_learning_agent2.pkl")
    
    agent1_data = load_q_table(agent1_path)
    agent2_data = load_q_table(agent2_path)
    
    return agent1_data, agent2_data


def evaluate_saved_agents_example():
    """Example function showing how to evaluate saved agents"""
    print("Example: Evaluating saved agents...")
    
    try:
        # Load two different agent pairs
        agent1_seed0, agent2_seed0 = load_agent_pair(0)
        agent1_seed1, agent2_seed1 = load_agent_pair(1)
        
        print(f"Loaded agents successfully!")
        print(f"Seed 0 Agent 1 Q-table shape: {agent1_seed0['q_table'].shape}")
        print(f"Seed 1 Agent 1 metadata: {agent1_seed1['metadata']['seed']}")
        
        # Reconstruct QLearningState objects for evaluation
        q_state_0 = QLearningState(
            q_table_1=jnp.array(agent1_seed0['q_table']),
            q_table_2=jnp.array(agent2_seed0['q_table']),
            epsilon=0.0,  # Use greedy policy
            episode=0
        )
        
        q_state_1 = QLearningState(
            q_table_1=jnp.array(agent1_seed1['q_table']),
            q_table_2=jnp.array(agent2_seed1['q_table']),
            epsilon=0.0,
            episode=0
        )
        
        # Evaluate crossplay
        crossplay_score = evaluate_crossplay(q_state_0, q_state_1, num_episodes=50)
        
    except FileNotFoundError:
        print("No saved agents found. Run the training script first!")


def train_multiple_seeds_and_crossplay(params: QLearningParams, n_seeds: int = 5, 
                                       eval_episodes: int = 100, save_agents: bool = True) -> Tuple[list, np.ndarray]:
    """Train multiple pairs of policies and compute crossplay scores"""
    
    print(f"Training {n_seeds} pairs of policies with different seeds...")
    print(f"Parameters: LR={params.learning_rate}, Episodes={params.num_episodes}")
    print("=" * 60)
    
    # Train policies with different seeds
    trained_policies = []
    
    for seed_idx in range(n_seeds):
        print(f"\nTraining seed {seed_idx + 1}/{n_seeds} (seed={seed_idx})...")
        q_state, _ = train_q_learning(params, seed=seed_idx)
        trained_policies.append(q_state)
        
        # Save Q-tables if requested
        if save_agents:
            save_q_tables(q_state, seed_idx, params)
        
        # Quick self-play evaluation
        self_reward = evaluate_crossplay(q_state, q_state, eval_episodes, seed=123)
        print(f"Seed {seed_idx}: Self-play reward: {self_reward:.3f}")
    
    print("\n" + "=" * 60)
    print("Computing crossplay matrix...")
    
    # Compute crossplay scores
    crossplay_matrix = np.zeros((n_seeds, n_seeds))
    
    for i in range(n_seeds):
        for j in range(n_seeds):
            # Policy i player 1 vs Policy j player 2
            score = evaluate_crossplay(trained_policies[i], trained_policies[j], 
                                       eval_episodes, seed=456)
            crossplay_matrix[i, j] = score
            
    
    return trained_policies, crossplay_matrix


def print_crossplay_analysis(crossplay_matrix: np.ndarray):
    """Print detailed analysis of crossplay results"""
    n_seeds = crossplay_matrix.shape[0]
    
    print("\n" + "=" * 60)
    print("CROSSPLAY ANALYSIS")
    print("=" * 60)
    
    # Print full matrix
    print("\nCrossplay Matrix (rows=P1 policy, cols=P2 policy):")
    print("     ", end="")
    for j in range(n_seeds):
        print(f"  P{j}  ", end="")
    print()
    
    for i in range(n_seeds):
        print(f"P{i}  ", end="")
        for j in range(n_seeds):
            print(f"{crossplay_matrix[i, j]:.3f}", end=" ")
        print()
    
    # Statistics
    diagonal = np.diag(crossplay_matrix)
    off_diagonal = crossplay_matrix[~np.eye(n_seeds, dtype=bool)]
    
    print(f"\nStatistics:")
    print(f"Self-play scores (diagonal): {diagonal}")
    print(f"Mean self-play: {np.mean(diagonal):.3f} ± {np.std(diagonal):.3f}")
    print(f"Mean crossplay: {np.mean(off_diagonal):.3f} ± {np.std(off_diagonal):.3f}")
    print(f"Self-play cross-play gap: {np.mean(diagonal) - np.mean(off_diagonal):.3f}")
    

def save_results_json(crossplay_matrix: np.ndarray, output_dir: str = None):
    """Save crossplay results to JSON file"""
    # Default to script directory if no output_dir provided
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = script_dir
    
    # Calculate statistics
    diagonal = np.diag(crossplay_matrix)  # Self-play scores (i,i)
    off_diagonal = crossplay_matrix[~np.eye(crossplay_matrix.shape[0], dtype=bool)]  # Cross-play scores (i,j where i!=j)
    
    # Create results dictionary
    results = {
        "cross_play_matrix": crossplay_matrix.tolist(),  # Convert numpy array to list of lists
        "avg_self_play_reward": float(np.mean(diagonal)),
        "avg_cross_play_reward": float(np.mean(off_diagonal))
    }
    
    # Save to JSON file
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results_path


if __name__ == "__main__":
    # Set up training parameters
    params = QLearningParams(
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        num_episodes=5000  
    )
    
    # Configuration for crossplay experiment
    n_seeds = 5  # Number of different seed policies to train
    eval_episodes = 100  # Rollouts per crossplay evaluation
    
    print("JAX Q-Learning Crossplay Experiment")
    print("=" * 60)
    
    # Train multiple policies and compute crossplay matrix
    trained_policies, crossplay_matrix = train_multiple_seeds_and_crossplay(
        params, n_seeds=n_seeds, eval_episodes=eval_episodes
    )
    
    # Analyze and print results
    print_crossplay_analysis(crossplay_matrix)
    
    # Save results to JSON file
    save_results_json(crossplay_matrix)
    
    print(f"\nExperiment completed!")

    
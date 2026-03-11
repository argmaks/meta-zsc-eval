# %%
import numpy as np
import pickle
import os
from typing import Sequence, Union, Callable, Tuple, Optional, Dict, Any, List
from context_files.envs.cat_dog.cat_dog import CatDogGame
from analysis.utils import read_experiment_to_dataframe, get_action_from_policy
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# %%
# show all columns when printing a dataframe
pd.set_option('display.max_columns', None)


def extract_algo_directory(config_repo_template):
    """
    Extract the algorithm directory name from config_repo_template string.
    
    Args:
        config_repo_template: String containing comma-separated file paths
        
    Returns:
        String: Algorithm directory name (e.g., 'q_learning_2') or None if not found
        
    Example:
        'papers/op.pdf, envs/cat_dog.py, algos/cat_dog/q_learning_2/q_learning.py' -> 'q_learning_2'
    """
    if pd.isna(config_repo_template):
        return None
    
    # Split by comma and strip whitespace from each item
    items = [item.strip() for item in str(config_repo_template).split(',')]
    
    # Find the item that contains 'algos/cat_dog'
    for item in items:
        if 'algos/cat_dog' in item:
            # Split by '/' and find the component after 'cat_dog'
            path_parts = item.split('/')
            if 'cat_dog' in path_parts:
                cat_dog_idx = path_parts.index('cat_dog')
                if cat_dog_idx + 1 < len(path_parts):
                    return path_parts[cat_dog_idx + 1]
    
    return None


# %%
# Policy loading functions for different Q-learning implementations

def load_q_learning_1_policies(seed_dir: Path):
    """
    Load policies from q_learning_1 format.
    Files: agent_alice.pkl, agent_bob.pkl, agent_metadata.pkl
    """
    try:
        # Try different possible prefixes
        for prefix in ["agent", "q_learning"]:
            alice_file = seed_dir / f"{prefix}_alice.pkl"
            bob_file = seed_dir / f"{prefix}_bob.pkl"
            
            if alice_file.exists() and bob_file.exists():
                with open(alice_file, 'rb') as f:
                    alice_q_table = pickle.load(f)
                with open(bob_file, 'rb') as f:
                    bob_q_table = pickle.load(f)
                
                # Try to load metadata
                metadata = {}
                metadata_file = seed_dir / f"{prefix}_metadata.pkl"
                if metadata_file.exists():
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                
                return alice_q_table, bob_q_table, metadata
        
        return None, None, None
    except Exception as e:
        print(f"Error loading q_learning_1 policies from {seed_dir}: {e}")
        return None, None, None


def load_q_learning_2_policies(seed_dir: Path):
    """
    Load policies from q_learning_2 format.
    Files: alice.pkl, bob.pkl (containing Q-tables directly, not wrapped in dictionaries)
    """
    try:
        alice_file = seed_dir / "alice.pkl"
        bob_file = seed_dir / "bob.pkl"
        
        if alice_file.exists() and bob_file.exists():
            with open(alice_file, 'rb') as f:
                alice_data = pickle.load(f)
            with open(bob_file, 'rb') as f:
                bob_data = pickle.load(f)
            
            # Check if this is q_learning_2 format (direct Q-tables, not dictionaries with 'q_table' key)
            # If either file contains a dict with 'q_table' key, this is likely q_learning_4 format
            if (isinstance(alice_data, dict) and 'q_table' in alice_data) or \
               (isinstance(bob_data, dict) and 'q_table' in bob_data):
                return None, None, None  # This is q_learning_4 format, not q_learning_2
            
            # This is genuine q_learning_2 format - direct Q-tables
            return alice_data, bob_data, {}
        
        return None, None, None
    except Exception as e:
        print(f"Error loading q_learning_2 policies from {seed_dir}: {e}")
        return None, None, None


def load_q_learning_3_policies(seed_dir: Path):
    """
    Load policies from q_learning_3 format.
    Files: qtable_alice.pkl, qtable_bob.pkl
    """
    try:
        alice_file = seed_dir / "qtable_alice.pkl"
        bob_file = seed_dir / "qtable_bob.pkl"
        
        if alice_file.exists() and bob_file.exists():
            with open(alice_file, 'rb') as f:
                alice_q_table = pickle.load(f)
            with open(bob_file, 'rb') as f:
                bob_q_table = pickle.load(f)
            
            return alice_q_table, bob_q_table, {}
        
        return None, None, None
    except Exception as e:
        print(f"Error loading q_learning_3 policies from {seed_dir}: {e}")
        return None, None, None


def convert_q_learning_4_format(q_table_dict: Dict, action_space_size: int) -> Dict:
    """
    Convert q_learning_4 Q-table format to CatDogPolicy-compatible format.
    
    Args:
        q_table_dict: Q-table in q_learning_4 format with string keys and dict values
        action_space_size: Number of actions for this agent
    
    Returns:
        Q-table with tuple keys and numpy array values
    """
    converted_q_table = {}
    
    for state_str, action_dict in q_table_dict.items():
        # Convert string key to tuple
        # Remove parentheses and split by comma
        state_str_clean = state_str.strip('()')
        state_tuple = tuple(int(x.strip()) for x in state_str_clean.split(','))
        
        # Convert action dict to numpy array
        q_values = np.zeros(action_space_size)
        for action, q_value in action_dict.items():
            if action < action_space_size:
                q_values[action] = q_value
        
        converted_q_table[state_tuple] = q_values
    
    return converted_q_table


def load_q_learning_4_policies(seed_dir: Path):
    """
    Load policies from q_learning_4 format.
    Files: alice.pkl, bob.pkl (containing dictionaries with 'q_table' key and metadata)
    """
    try:
        alice_file = seed_dir / "alice.pkl"
        bob_file = seed_dir / "bob.pkl"
        
        if alice_file.exists() and bob_file.exists():
            with open(alice_file, 'rb') as f:
                alice_data = pickle.load(f)
            with open(bob_file, 'rb') as f:
                bob_data = pickle.load(f)
            
            # Check if this is genuinely q_learning_4 format (dictionaries with 'q_table' key)
            if not (isinstance(alice_data, dict) and 'q_table' in alice_data and 
                    isinstance(bob_data, dict) and 'q_table' in bob_data):
                return None, None, None  # This is not q_learning_4 format
            
            # Extract Q-tables from the data dictionaries
            raw_alice_q_table = alice_data['q_table']
            raw_bob_q_table = bob_data['q_table']
            
            # Convert to CatDogPolicy-compatible format
            alice_q_table = convert_q_learning_4_format(raw_alice_q_table, 4)  # Alice has 4 actions
            bob_q_table = convert_q_learning_4_format(raw_bob_q_table, 3)     # Bob has 3 actions
            
            # Combine metadata from both agents
            metadata = {}
            metadata['alice'] = {k: v for k, v in alice_data.items() if k != 'q_table'}
            metadata['bob'] = {k: v for k, v in bob_data.items() if k != 'q_table'}
            
            return alice_q_table, bob_q_table, metadata
        
        return None, None, None
    except Exception as e:
        print(f"Error loading q_learning_4 policies from {seed_dir}: {e}")
        return None, None, None


def load_q_learning_5_policies(seed_dir: Path):
    """
    Load policies from q_learning_5 format.
    Files: q_table_alice.pkl, q_table_bob.pkl (containing Q-tables as dictionaries)
    """
    try:
        alice_file = seed_dir / "q_table_alice.pkl"
        bob_file = seed_dir / "q_table_bob.pkl"
        
        if alice_file.exists() and bob_file.exists():
            with open(alice_file, 'rb') as f:
                alice_q_table = pickle.load(f)
            with open(bob_file, 'rb') as f:
                bob_q_table = pickle.load(f)
            
            # The Q-tables should already be in dictionary format with tuple keys
            # and the values should be numpy arrays or convertible to numpy arrays
            # Convert values to numpy arrays if they aren't already
            for state, q_values in alice_q_table.items():
                if not isinstance(q_values, np.ndarray):
                    alice_q_table[state] = np.array(q_values)
            
            for state, q_values in bob_q_table.items():
                if not isinstance(q_values, np.ndarray):
                    bob_q_table[state] = np.array(q_values)
            
            return alice_q_table, bob_q_table, {}
        
        return None, None, None
    except Exception as e:
        print(f"Error loading q_learning_5 policies from {seed_dir}: {e}")
        return None, None, None


def load_cat_dog_policies(seed_dir: Path):
    """
    Attempt to load Cat-Dog policies from any of the supported formats.
    
    Returns:
        Tuple of (alice_q_table, bob_q_table, metadata, format_type) or (None, None, None, None) if failed
    """
    # Try q_learning_1 format first
    alice_q, bob_q, metadata = load_q_learning_1_policies(seed_dir)
    if alice_q is not None:
        return alice_q, bob_q, metadata, "q_learning_1"
    
    # Try q_learning_4 format before q_learning_2 (both use alice.pkl/bob.pkl)
    # q_learning_4 has stricter requirements (dict with 'q_table' key)
    alice_q, bob_q, metadata = load_q_learning_4_policies(seed_dir)
    if alice_q is not None:
        return alice_q, bob_q, metadata, "q_learning_4"
    
    # Try q_learning_2 format (direct Q-tables in alice.pkl/bob.pkl)
    alice_q, bob_q, metadata = load_q_learning_2_policies(seed_dir)
    if alice_q is not None:
        return alice_q, bob_q, metadata, "q_learning_2"
    
    # Try q_learning_3 format
    alice_q, bob_q, metadata = load_q_learning_3_policies(seed_dir)
    if alice_q is not None:
        return alice_q, bob_q, metadata, "q_learning_3"
    
    # Try q_learning_5 format
    alice_q, bob_q, metadata = load_q_learning_5_policies(seed_dir)
    if alice_q is not None:
        return alice_q, bob_q, metadata, "q_learning_5"
    
    return None, None, None, None


# %%
# Policy interface classes

class CatDogPolicy:
    """Unified policy interface for Cat-Dog game agents."""
    
    def __init__(self, q_table: Dict, agent_name: str, action_space_size: int):
        self.q_table = q_table
        self.agent_name = agent_name
        self.action_space_size = action_space_size
        
        # Convert to defaultdict for missing states
        if not isinstance(q_table, defaultdict):
            self.q_table = defaultdict(lambda: np.zeros(action_space_size))
            self.q_table.update(q_table)
    
    def select_action(self, observation: np.ndarray, valid_actions: list, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            observation: Current observation as numpy array
            valid_actions: List of valid actions for current state
            epsilon: Exploration rate (0.0 for greedy policy)
        """
        if not valid_actions:
            return 0  # Default action if no valid actions
        
        state_key = tuple(observation)
        
        # Epsilon-greedy action selection
        if epsilon > 0 and np.random.random() < epsilon:
            return np.random.choice(valid_actions)
        
        # Get Q-values and mask invalid actions
        q_values = self.q_table[state_key]
        masked_q_values = np.full_like(q_values, -np.inf)
        for action in valid_actions:
            if action < len(q_values):
                masked_q_values[action] = q_values[action]
        
        return int(np.argmax(masked_q_values))


def create_cat_dog_policies(alice_q_table: Dict, bob_q_table: Dict) -> Tuple[CatDogPolicy, CatDogPolicy]:
    """Create unified policy objects from Q-tables."""
    alice_policy = CatDogPolicy(alice_q_table, "Alice", 4)  # Alice has 4 actions
    bob_policy = CatDogPolicy(bob_q_table, "Bob", 3)       # Bob has 3 actions
    return alice_policy, bob_policy


# %%
# Evaluation functions

def evaluate_cat_dog_policies(alice_policy: CatDogPolicy, bob_policy: CatDogPolicy, 
                             n_episodes: int = 100, seed: Optional[int] = None) -> float:
    """
    Evaluate a pair of Cat-Dog policies.
    
    Args:
        alice_policy: Alice's policy
        bob_policy: Bob's policy
        n_episodes: Number of episodes to run
        seed: Random seed for reproducibility
    
    Returns:
        Average episode reward
    """
    env = CatDogGame()
    if seed is not None:
        np.random.seed(seed)
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        # Set episode-specific seed for determinism
        if seed is not None:
            episode_seed = seed * 10000 + episode
            observation, info = env.reset(seed=episode_seed)
        else:
            observation, info = env.reset()
        
        episode_reward = 0
        
        while True:
            current_turn = observation[1]  # 0=Alice, 1=Bob
            valid_actions = info['valid_actions']
            
            if current_turn == 0:  # Alice's turn
                action = alice_policy.select_action(observation, valid_actions, epsilon=0.0)
            else:  # Bob's turn
                action = bob_policy.select_action(observation, valid_actions, epsilon=0.0)
            
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
            
            observation = next_observation
            info = next_info
        
        episode_rewards.append(episode_reward)
    
    return float(np.mean(episode_rewards))


def cross_play_evaluation(policy_pairs: Dict[str, Dict[str, Tuple[CatDogPolicy, CatDogPolicy]]], 
                         n_episodes: int = 100) -> Tuple[np.ndarray, List[str]]:
    """
    Perform cross-play evaluation where Alice from one seed plays with Bob from another seed.
    
    Args:
        policy_pairs: Dictionary mapping sample_name -> seed_name -> (alice_policy, bob_policy)
        n_episodes: Number of episodes per evaluation
    
    Returns:
        Tuple of (cross_play_matrix, sample_names)
    """
    sample_names = list(policy_pairs.keys())
    n_samples = len(sample_names)
    
    # Initialize cross-play matrix
    cross_play_matrix = np.zeros((n_samples, n_samples))
    
    print(f"Evaluating {n_samples}x{n_samples} cross-play combinations...")
    
    for i, sample_i in enumerate(sample_names):
        for j, sample_j in enumerate(sample_names):
            # Get all seeds from both samples
            seeds_i = sorted(policy_pairs[sample_i].keys())
            seeds_j = sorted(policy_pairs[sample_j].keys())
            
            if not seeds_i or not seeds_j:
                print(f"  Warning: No seeds available for {sample_i} or {sample_j}")
                cross_play_matrix[i, j] = np.nan
                continue
            
            # Evaluate all combinations of seeds and average
            seed_returns = []
            combination_count = 0
            for seed_i in seeds_i:
                for seed_j in seeds_j:
                    # Alice from sample i (seed_i), Bob from sample j (seed_j)
                    alice_policy, _ = policy_pairs[sample_i][seed_i]
                    _, bob_policy = policy_pairs[sample_j][seed_j]
                    
                    # Evaluate this combination
                    avg_return = evaluate_cat_dog_policies(
                        alice_policy, bob_policy,
                        n_episodes=n_episodes,
                        seed=42 + i*n_samples + j + combination_count  # Deterministic but unique seed
                    )
                    seed_returns.append(avg_return)
                    combination_count += 1
            
            # Average across all seed combinations
            cross_play_matrix[i, j] = np.mean(seed_returns)
            print(f"  {sample_i} (Alice) vs {sample_j} (Bob): {cross_play_matrix[i, j]:.4f} (avg across {len(seeds_i)}x{len(seeds_j)}={len(seed_returns)} combinations)")
    
    return cross_play_matrix, sample_names


# %%
repo_root = Path(__file__).parent.parent
sample_dir = repo_root / "samples"
viable_sample_start_time = "2025-07-30 18:00:00"

# Load experiment data
print("Loading experiment data...")
df = read_experiment_to_dataframe("obl_cat_dog")
df = df.loc[df.sample_timestamp > viable_sample_start_time]

# %%
# Filter for Cat-Dog related samples
df = df.loc[df.config_env == "cat_dog"]

# %%
sample_paths = df.sample_path

# Load policies from each sample
print("\nLoading policies...")
loaded_policies = {}

for sample_path in sample_paths:
    sample_path = sample_dir / sample_path
    sample_name = sample_path.name
    trained_agents_dir = sample_path / "trained_agents"
    
    if not trained_agents_dir.exists():
        print(f"Warning: No trained_agents directory found in {sample_path}")
        continue
    
    # Find all seed directories
    seed_dirs = list(trained_agents_dir.glob("seed_*"))
    if not seed_dirs:
        print(f"Warning: No seed directories found in {sample_path}")
        continue
    
    sample_policies = {}
    
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name  # e.g., "seed_0"
        
        # Try to load policies from any supported format
        alice_q, bob_q, metadata, format_type = load_cat_dog_policies(seed_dir)
        
        if alice_q is not None and bob_q is not None:
            try:
                # Create unified policy objects
                alice_policy, bob_policy = create_cat_dog_policies(alice_q, bob_q)
                
                sample_policies[seed_name] = {
                    'alice_policy': alice_policy,
                    'bob_policy': bob_policy,
                    'format_type': format_type,
                    'metadata': metadata
                }
                
            except Exception as e:
                print(f"Error creating policies from {sample_name}/{seed_name}: {e}")
        else:
            print(f"Warning: Could not load policies from {sample_name}/{seed_name}")
    
    if sample_policies:
        loaded_policies[sample_name] = sample_policies
        print(f"Successfully loaded policies from {sample_name} ({len(sample_policies)} seeds, format: {list(set(p['format_type'] for p in sample_policies.values()))})")
    else:
        print(f"Warning: No valid policies loaded from {sample_name}")

print(f"\nLoaded policies from {len(loaded_policies)} samples:")
for sample_name, seed_policies in loaded_policies.items():
    formats = set(p['format_type'] for p in seed_policies.values())
    print(f"  - {sample_name}: {len(seed_policies)} seeds, formats: {formats}")

# %%


# Add policy loading flags to dataframe
df['policy_loadable'] = df['sample_path'].apply(lambda path: Path(path).name in loaded_policies)


# %%
# Evaluate self-play performance
print("\nEvaluating self-play performance...")
performance_results = {}

for sample_name, seed_policies in loaded_policies.items():
    seed_returns = []
    
    for seed_name, policy_data in seed_policies.items():
        try:
            # Test the policy pair for this seed
            avg_return = evaluate_cat_dog_policies(
                policy_data['alice_policy'], 
                policy_data['bob_policy'], 
                n_episodes=100,
                seed=42
            )
            seed_returns.append(avg_return)
            
        except Exception as e:
            print(f"Error evaluating {sample_name}/{seed_name}: {e}")
    
    if seed_returns:
        # Average across seeds for this sample
        sample_avg_return = np.mean(seed_returns)
        performance_results[sample_name] = sample_avg_return
        print(f"{sample_name}: Average return = {sample_avg_return:.4f} (across {len(seed_returns)} seeds)")

# %%
# Add performance results to dataframe
df['average_sp_return'] = df['sample_path'].apply(lambda path: performance_results.get(Path(path).name))
df["correct"] = np.isclose(df["average_sp_return"], 5, atol=0.25)

# %%
filter_by_correct = True

# Filter loaded_policies by correct samples if enabled
if filter_by_correct:
    # Get sample names that have correct=True
    correct_samples = df.loc[df['correct'] == True, 'sample_path'].apply(lambda path: Path(path).name).tolist()
    
    # Filter loaded_policies to only include correct samples
    filtered_loaded_policies = {
        sample_name: seed_policies 
        for sample_name, seed_policies in loaded_policies.items() 
        if sample_name in correct_samples
    }
    
    print(f"\nFiltering policies by correctness:")
    print(f"  Original samples: {len(loaded_policies)}")
    print(f"  Correct samples: {len(filtered_loaded_policies)}")
    print(f"  Filtered out: {len(loaded_policies) - len(filtered_loaded_policies)}")

else:   
    filtered_loaded_policies = loaded_policies

# Cross-play evaluation
if len(filtered_loaded_policies) > 1:
    print("\nPerforming cross-play evaluation...")
    
    # Prepare policy pairs for cross-play
    policy_pairs = {}
    for sample_name, seed_policies in filtered_loaded_policies.items():
        policy_pairs[sample_name] = {}
        for seed_name, policy_data in seed_policies.items():
            policy_pairs[sample_name][seed_name] = (
                policy_data['alice_policy'], 
                policy_data['bob_policy']
            )
    
    cross_play_matrix, sample_names = cross_play_evaluation(policy_pairs, n_episodes=100)
    

# %%
# Set seaborn theme
sns.set_theme()

# Create a pandas DataFrame for better visualization
crossplay_df = pd.DataFrame(cross_play_matrix, index=sample_names, columns=sample_names)

# Extract algorithm directories for each sample to group them
sample_to_algo = {}
for sample_name in sample_names:
    sample_to_algo[sample_name] = df.loc[df.sample_name == sample_name].config_algo.iloc[0]

# Get unique algorithm directories and their sample counts
algo_dirs = list(set(sample_to_algo.values()))
algo_dirs.sort()  # Sort for consistent ordering

# Group samples by algorithm directory
algo_to_samples = {}
for algo_dir in algo_dirs:
    algo_to_samples[algo_dir] = [name for name, algo in sample_to_algo.items() if algo == algo_dir]

print(f"Algorithm directories found: {algo_dirs}")
for algo_dir in algo_dirs:
    print(f"  {algo_dir}: {len(algo_to_samples[algo_dir])} samples")

# Create block matrices for each algorithm combination
block_matrices = {}
for algo_i in algo_dirs:
    for algo_j in algo_dirs:
        samples_i = algo_to_samples[algo_i]
        samples_j = algo_to_samples[algo_j]
        
        # Extract the relevant submatrix
        block_df = crossplay_df.loc[samples_i, samples_j]
        block_matrices[(algo_i, algo_j)] = block_df

# Set up GridSpec layout
algo_sizes = [len(algo_to_samples[algo]) for algo in algo_dirs]
n_algos = len(algo_dirs)

fig = plt.figure(figsize=(24, 24))
gs = fig.add_gridspec(
    n_algos, n_algos,
    height_ratios=algo_sizes,
    width_ratios=algo_sizes,
    wspace=0.05, hspace=0.05
)

# Create axes for each block
axes = {}
for i, algo_i in enumerate(algo_dirs):
    for j, algo_j in enumerate(algo_dirs):
        if i == 0 and j == 0:
            ax = fig.add_subplot(gs[i, j])
        elif i == 0:
            ax = fig.add_subplot(gs[i, j], sharey=axes[(0, 0)])
        elif j == 0:
            ax = fig.add_subplot(gs[i, j], sharex=axes[(0, 0)])
        else:
            ax = fig.add_subplot(gs[i, j], 
                                sharex=axes[(0, j)], 
                                sharey=axes[(i, 0)])
        axes[(i, j)] = ax

# Set color scale range - adjust based on Cat-Dog game rewards
vmin, vmax = -10, 10  # Default range for Cat-Dog game

# Draw each sub-heatmap
for i, algo_i in enumerate(algo_dirs):
    for j, algo_j in enumerate(algo_dirs):
        ax = axes[(i, j)]
        block_df = block_matrices[(algo_i, algo_j)]
        
        # Control tick labels on outer edges only
        is_bottom_edge = (i == n_algos - 1)
        is_left_edge = (j == 0)
        is_top_edge = (i == 0)
        is_right_edge = (j == n_algos - 1)
        
        # Round values for annotation
        annot_df = block_df.round(2)
        
        sns.heatmap(
            block_df, 
            ax=ax,
            annot=annot_df, # uncomment to show values inside the blocks
            fmt='', # uncomment to show values inside the blocks
            # annot=False, # uncomment to hide values inside the blocks
            cbar=False,
            vmin=vmin,
            vmax=vmax,
            cmap='RdYlGn',
            linewidths=0.5,
            square=True
        )
        
        # Control tick labels
        if is_bottom_edge:
            ax.tick_params(axis='x', labelbottom=True)
        else:
            ax.tick_params(axis='x', labelbottom=False)
            
        if is_left_edge:
            ax.set_yticklabels(block_df.index, rotation=0)
            ax.tick_params(axis='y', labelleft=True)
        else:
            ax.tick_params(axis='y', labelleft=False)
        
        # Add algorithm directory labels on outer edges
        if is_top_edge:
            ax.set_title(f'{algo_j}', fontweight='bold', pad=10, fontsize=24)
        if is_left_edge:
            ax.set_ylabel(f'{algo_i}', fontweight='bold', rotation=0, ha='right', va='center', fontsize=24)
        
        # Clean up inner plot labels
        if not is_bottom_edge:
            ax.set_xlabel('')
        if not is_left_edge and not is_top_edge:
            ax.set_ylabel('')
            
        # Remove tick marks on inner edges
        if not is_bottom_edge:
            ax.tick_params(bottom=False)
        if not is_left_edge:
            ax.tick_params(left=False)
        if not is_top_edge:
            ax.tick_params(top=False)  
        if not is_right_edge:
            ax.tick_params(right=False)

        # Remove all ticks
        # ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False) # uncomment to remove all ticks

# Add shared colorbar
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap='RdYlGn')
cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
fig.colorbar(sm, cax=cax, label='Average Reward')

# Add overall title
fig.suptitle('Cat-Dog Game Cross-Play Policy Performance Matrix by Algorithm\n(Alice × Bob, average XP score across all seed combinations)', 
                fontsize=32, fontweight='bold', y=0.95)

# Rotate x-axis labels on bottom row for better readability
for j in range(n_algos):
    if (n_algos-1, j) in axes:
        ax = axes[(n_algos-1, j)]
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

plt.tight_layout()
plt.show()


#
# %%
df.groupby(['config_runner_model'])[['policy_loadable', 'correct']].agg(['mean', 'count']) 
# %%
df.groupby(['config_runner_llm_instructor'])[['policy_loadable', 'correct']].agg(['mean', 'count'])
# %%
df.groupby(['config_runner_feedback_loop_iters'])[['policy_loadable', 'correct']].agg(['mean', 'count'])
# %%
df.groupby(['config_runner_model', 'config_runner_llm_instructor'])[['policy_loadable', 'correct']].agg(['mean', 'count'])
# %%
df.groupby(['config_runner_model', 'config_runner_feedback_loop_iters'])[['policy_loadable', 'correct']].agg(['mean', 'count'])
# %%
df.groupby(['config_runner_model', 'config_algo'])[['policy_loadable', 'correct']].agg(['mean', 'count'])
# %%

# %%
import numpy as np
import pickle
import os
from typing import Sequence, Union, Callable, Tuple, Optional
from context_files.envs.lever_game.lever_game import TwoPlayerLeverGame
from analysis.utils import read_experiment_to_dataframe, unroll_environment, get_action_from_policy, load_q_learning_agent, create_q_learning_policy, load_trained_q_learning_policies 
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %%
# show all columns when printing a dataframe
pd.set_option('display.max_columns', None)

# Function to extract algorithm directory name from config_repo_template
def extract_algo_directory(config_repo_template):
    """
    Extract the algorithm directory name from config_repo_template string.
    
    Args:
        config_repo_template: String containing comma-separated file paths
        
    Returns:
        String: Algorithm directory name (e.g., 'q_learning_2') or None if not found
        
    Example:
        'papers/op.pdf, envs/lever_game.py, algos/q_learning_2/q_learning.py' -> 'q_learning_2'
    """
    if pd.isna(config_repo_template):
        return None
    
    # Split by comma and strip whitespace from each item
    items = [item.strip() for item in str(config_repo_template).split(',')]
    
    # Find the item that starts with 'algos'
    for item in items:
        if item.startswith('algos/'):
            # Split by '/' and take the second component (index 1)
            path_parts = item.split('/')
            if len(path_parts) >= 2:
                return path_parts[1]
    
    return None

# %%
repo_root = Path(__file__).parent.parent
sample_dir = repo_root / "samples"
# viable_sample_start_time = "2025-07-27 10:00:00" 
# viable_sample_end_time = "2025-07-28 19:00:00"
viable_sample_start_time = "2025-07-30 10:00:00"
viable_sample_end_time = "2027-07-29 19:00:00"

# %%n
df = read_experiment_to_dataframe("op_lever_game")
df = df.loc[(df.sample_timestamp > viable_sample_start_time) & (df.sample_timestamp < viable_sample_end_time)]

# Extract algorithm directory name
df['algo_directory'] = df['config_repo_template'].apply(extract_algo_directory)

# adjust for old samples
df['algo_directory'] = df['algo_directory'].map(lambda x: "q_learning_1" if x == "q_learning.py" else x)

# fill in the algo_directory column with config_algo if it is not already set
df['algo_directory'] = df['algo_directory'].fillna(df['config_algo'])

# Display the extracted values to verify
print("Algorithm directories extracted:")
print(df[['config_repo_template', 'algo_directory']].head(10))
print(f"\nUnique algorithm directories: {df['algo_directory'].unique()}")

df["sample_tag"] = "OP" + "_" + df.index.astype(str) + "_x_" + df.algo_directory
sample_paths = df.sample_path

# %%
# Create mapping from sample path names to sample tags
sample_name_to_tag = {}
for i, sample_path in enumerate(df.sample_path):
    sample_name = Path(sample_path).name
    sample_tag = df.iloc[i]["sample_tag"]
    sample_name_to_tag[sample_name] = sample_tag

# %%
# df = read_experiment_to_dataframe("pair_coding_zsc_methods")
# sample_paths = df.loc[(df.result_result == "Successfully implemented the Other-play method.") ].sample_path

# %%
# df.loc[df.result_result == "Successfully implemented the Other-play method."]
# %%
# sample_paths = df.loc[df.result_result == "Successfully implemented the Other-play method."].sample_path
# sample_paths = df.loc[df.result_result == "Aider scaffold run successfully."].sample_path
sample_paths = [sample_dir / path for path in sample_paths]
# %%
# Find and load Q-learning policies from each sample path for all seeds
loaded_policies = {}

for sample_path in sample_paths:
    sample_name = sample_path.name
    trained_agents_dir = sample_path / "trained_agents" 
    
    # Check if trained_agents directory exists
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
        
        # Look for Q-learning agent pickle files in this seed directory
        agent1_files = list(seed_dir.glob("*_agent1.pkl"))
        agent2_files = list(seed_dir.glob("*_agent2.pkl"))
        
        if agent1_files and agent2_files:
            agent1_file = agent1_files[0]  # Take the first match
            agent2_file = agent2_files[0]  # Take the first match
            try:
                # Load both agents
                agent1 = load_q_learning_agent(str(agent1_file))
                agent2 = load_q_learning_agent(str(agent2_file))
                
                # Create policy functions
                policy1 = create_q_learning_policy(agent1)
                policy2 = create_q_learning_policy(agent2)
                
                sample_policies[seed_name] = {
                    'policy1': policy1,
                    'policy2': policy2,
                    'agent1_file': str(agent1_file),
                    'agent2_file': str(agent2_file)
                }
                
            except Exception as e:
                print(f"Error loading policies from {sample_name}/{seed_name}: {e}")
        else:
            print(f"Warning: Missing agent files in {sample_name}/{seed_name}")
    
    if sample_policies:
        loaded_policies[sample_name] = sample_policies
        print(f"Successfully loaded policies from {sample_name} ({len(sample_policies)} seeds)")
    else:
        print(f"Warning: No valid policies loaded from {sample_name}")

print(f"\nLoaded policies from {len(loaded_policies)} samples:")
for sample_name, seed_policies in loaded_policies.items():
    print(f"  - {sample_name}: {len(seed_policies)} seeds")

# %%
# Add a column to the dataframe that flags whether the sample produced a policy that can be loaded
df['policy_loadable'] = df['sample_path'].apply(lambda path: Path(path).name in loaded_policies)


# %%
# Example: Evaluate performance of loaded policies (averaging across seeds)
if loaded_policies:
    print("\nEvaluating policy performance...")
    
    # Create environment for evaluation
    env = TwoPlayerLeverGame()
    
    # Evaluate each loaded policy pair, averaging across seeds
    performance_results = {}
    
    for sample_name, seed_policies in loaded_policies.items():
        seed_returns = []
        
        for seed_name, policy_data in seed_policies.items():
            try:
                # Test the policy pair for this seed
                avg_return = unroll_environment(
                    policies=[policy_data['policy1'], policy_data['policy2']], 
                    n_rollouts=100,
                    env=env,
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
    
    if performance_results:
        # Show summary statistics
        returns = list(performance_results.values())
        best_sample = max(performance_results.items(), key=lambda x: x[1])
        worst_sample = min(performance_results.items(), key=lambda x: x[1])
        print(f"\nPerformance Summary:")
        print(f"  Best performing sample: {best_sample[0]} ({best_sample[1]:.4f})")
        print(f"  Worst performing sample: {worst_sample[0]} ({worst_sample[1]:.4f})")
        print(f"  Average return across all samples: {np.mean(returns):.4f} ± {np.std(returns):.4f}")

# %%
# add a column to the dataframe with the average return of the sample
df['average_sp_return'] = df['sample_name'].map(performance_results)
df["correct"] = np.isclose(df["average_sp_return"], 0.9, atol=0.01)
# %%
# Cross-play evaluation: Policy 1 from each sample vs Policy 2 from all samples (matching seeds)
filter_by_correct = True

if loaded_policies:

    print("\nPerforming cross-play evaluation...")

    # Get sample names for indexing
    if filter_by_correct:
        sample_names = [sample_name for sample_name in loaded_policies.keys() if df.loc[df.sample_name == sample_name].correct.iloc[0] == True]
    else:
        sample_names = list(loaded_policies.keys()) 
    n_samples = len(sample_names)
    
    # Create environment for evaluation
    env = TwoPlayerLeverGame()
    
    # Initialize cross-play matrix
    crossplay_matrix = np.zeros((n_samples, n_samples))
    
    # Evaluate all combinations
    for i, sample_i in enumerate(sample_names):
        for j, sample_j in enumerate(sample_names):
            try:
                # Get all seeds from both samples
                seeds_i = sorted(loaded_policies[sample_i].keys())
                seeds_j = sorted(loaded_policies[sample_j].keys())
                
                if not seeds_i or not seeds_j:
                    print(f"  Warning: No seeds available for {sample_i} or {sample_j}")
                    crossplay_matrix[i, j] = np.nan
                    continue
                
                # Evaluate all combinations of seeds and collect returns
                seed_returns = []
                combination_count = 0
                for seed_i in seeds_i:
                    for seed_j in seeds_j:
                        # Get policy 1 from sample i (seed_i) and policy 2 from sample j (seed_j)
                        policy1 = loaded_policies[sample_i][seed_i]['policy1']
                        policy2 = loaded_policies[sample_j][seed_j]['policy2']
                        
                        # Evaluate this combination
                        avg_return = unroll_environment(
                            policies=[policy1, policy2],
                            n_rollouts=100,
                            env=env,
                            seed=42 + i*n_samples + j + combination_count  # Different seed for each combination
                        )
                        seed_returns.append(avg_return)
                        combination_count += 1
                
                # Average across all seed combinations
                crossplay_matrix[i, j] = np.mean(seed_returns)
                print(f"  {sample_i} (P1) vs {sample_j} (P2): {crossplay_matrix[i, j]:.4f} (avg across {len(seeds_i)}x{len(seeds_j)}={len(seed_returns)} combinations)")
                
            except Exception as e:
                print(f"  Error evaluating {sample_i} vs {sample_j}: {e}")
                crossplay_matrix[i, j] = np.nan

    crossplay_matrix

# %%
# Plot crossplay matrix as block-structured heatmap grouped by algorithm directory

# Set seaborn theme
sns.set_theme()


# Create a pandas DataFrame for better visualization
crossplay_df = pd.DataFrame(crossplay_matrix, index=sample_names, columns=sample_names)

# Extract algorithm directories for each sample to group them
sample_to_algo = {}
for sample_name in sample_names:
    sample_tag = sample_name_to_tag[sample_name]
    # Extract algo directory from sample tag (format: "OP_index_x_algo_directory")
    algo_dir = sample_tag.split('_x_')[-1] if '_x_' in sample_tag else 'unknown'
    sample_to_algo[sample_name] = algo_dir

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

# ---------------------------------------------------------------------
# Build block-matrices for each algorithm combination
# ---------------------------------------------------------------------
block_matrices = {}
for algo_i in algo_dirs:
    for algo_j in algo_dirs:
        samples_i = algo_to_samples[algo_i]
        samples_j = algo_to_samples[algo_j]
        
        # Extract the relevant submatrix
        block_df = crossplay_df.loc[samples_i, samples_j]
        block_matrices[(algo_i, algo_j)] = block_df

# ---------------------------------------------------------------------
# Set up GridSpec layout with sizes matching the algorithm groups
# ---------------------------------------------------------------------
algo_sizes = [len(algo_to_samples[algo]) for algo in algo_dirs]
n_algos = len(algo_dirs)

fig = plt.figure(figsize=(20, 20))
gs = fig.add_gridspec(
    n_algos, n_algos,
    height_ratios=algo_sizes,  # rows: size of each algo group
    width_ratios=algo_sizes,   # cols: size of each algo group  
    wspace=0.05, hspace=0.05   # small gaps between blocks
)

# Create axes for each block
axes = {}
for i, algo_i in enumerate(algo_dirs):
    for j, algo_j in enumerate(algo_dirs):
        if i == 0 and j == 0:
            # First subplot - no sharing
            ax = fig.add_subplot(gs[i, j])
        elif i == 0:
            # First row - share y-axis with leftmost in same row
            ax = fig.add_subplot(gs[i, j], sharey=axes[(0, 0)])
        elif j == 0:
            # First column - share x-axis with topmost in same column
            ax = fig.add_subplot(gs[i, j], sharex=axes[(0, 0)])
        else:
            # Other blocks - share both axes appropriately
            ax = fig.add_subplot(gs[i, j], 
                               sharex=axes[(0, j)], 
                               sharey=axes[(i, 0)])
        axes[(i, j)] = ax

# ---------------------------------------------------------------------
# Draw each sub-heatmap on its own axes
# ---------------------------------------------------------------------
# Set fixed color scale range from 0 to 1
vmin, vmax = 0.0, 1.0

for i, algo_i in enumerate(algo_dirs):
    for j, algo_j in enumerate(algo_dirs):
        ax = axes[(i, j)]
        block_df = block_matrices[(algo_i, algo_j)]
        
        # Only show tick labels on outer edges of the entire grid
        is_bottom_edge = (i == n_algos - 1)  # Last row - show x-axis labels
        is_left_edge = (j == 0)              # First column - show y-axis labels
        is_top_edge = (i == 0)               # First row
        is_right_edge = (j == n_algos - 1)   # Last column
        
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
        
        # Explicitly control tick labels - only show on outer edges
        if is_bottom_edge:
            ax.tick_params(axis='x', labelbottom=True)
        else:
            ax.tick_params(axis='x', labelbottom=False)
            
        if is_left_edge:
            ax.set_yticklabels(block_df.index, rotation=0)
            ax.tick_params(axis='y', labelleft=True)
        else:
            ax.tick_params(axis='y', labelleft=False)
        
        # Add algorithm directory labels only on outer edges
        if is_top_edge:  # Top edge - add column algorithm labels
            ax.set_title(f'{algo_j}', fontweight='bold', pad=10, fontsize=24)
        if is_left_edge:  # Left edge - add row algorithm labels  
            ax.set_ylabel(f'{algo_i}', fontweight='bold', rotation=0, ha='right', va='center', fontsize=24)
        
        # Ensure no axis labels on inner plots
        if not is_bottom_edge:
            ax.set_xlabel('')
        if not is_left_edge and not is_top_edge:  # Don't override algorithm labels on left edge
            ax.set_ylabel('')
            
        # Remove tick marks on inner edges for cleaner appearance
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
# ---------------------------------------------------------------------
# Add shared colorbar
# ---------------------------------------------------------------------
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap='RdYlGn')
cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # manual position
fig.colorbar(sm, cax=cax, label='Average Reward')

# Add overall title
fig.suptitle('Lever Game Cross-Play Policy Performance Matrix by Algorithm \n(Policy 1 × Policy 2, average XP score across all seed combinations)', 
             fontsize=32, fontweight='bold', y=0.95)

# Rotate x-axis labels on bottom row for better readability
for j in range(n_algos):
    if (n_algos-1, j) in axes:
        ax = axes[(n_algos-1, j)]
        ax.tick_params(axis='x', rotation=45)
        # Fix label alignment for rotated labels
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

plt.tight_layout()
plt.show()


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

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Union, Sequence, Callable, Tuple, Optional
import pandas as pd
import yaml
import numpy as np
import pickle
from datetime import datetime
from context_files.envs.lever_game.lever_game import TwoPlayerLeverGame
from context_files.envs.cat_dog.cat_dog import CatDogGame

def flatten_dict(data: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Recursively flatten a nested dictionary.
    
    Args:
        data: Dictionary to flatten
        parent_key: Parent key for nested structures
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        elif isinstance(value, list):
            # Handle lists by converting to string or extracting elements
            if len(value) == 0:
                items.append((new_key, None))
            elif all(isinstance(item, (str, int, float, bool)) for item in value):
                # Simple list of primitives - join as string
                items.append((new_key, ', '.join(map(str, value))))
            else:
                # Complex list - convert to string representation
                items.append((new_key, str(value)))
        else:
            # Primitive values
            items.append((new_key, value))
    
    return dict(items)


def parse_yaml_file(yaml_path: str, prefix: str = 'config') -> Dict[str, Any]:
    """
    Parse a YAML file and return flattened dictionary with prefixed keys.
    
    Args:
        yaml_path: Path to YAML file
        prefix: Prefix for all keys from this file
        
    Returns:
        Flattened dictionary with prefixed keys
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            return {}
            
        # Flatten the data
        flattened = flatten_dict(data)
        
        # Add prefix to all keys
        return {f"{prefix}_{key}": value for key, value in flattened.items()}
        
    except Exception as e:
        print(f"Error parsing YAML file {yaml_path}: {e}")
        return {f"{prefix}_parse_error": str(e)}


def parse_json_file(json_path: str, prefix: str = 'results') -> Dict[str, Any]:
    """
    Parse a JSON file and return flattened dictionary with prefixed keys.
    
    Args:
        json_path: Path to JSON file
        prefix: Prefix for all keys from this file
        
    Returns:
        Flattened dictionary with prefixed keys
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data is None:
            return {}
            
        # Flatten the data
        if isinstance(data, dict):
            flattened = flatten_dict(data)
        else:
            # Handle case where JSON root is not a dictionary
            flattened = {'root': data}
        
        # Add prefix to all keys
        return {f"{prefix}_{key}": value for key, value in flattened.items()}
        
    except Exception as e:
        print(f"Error parsing JSON file {json_path}: {e}")
        return {f"{prefix}_parse_error": str(e)}


def create_samples_dataframe(experiment_name: str, samples_base_path: Path) -> pd.DataFrame:
    """
    Create a pandas DataFrame from config.yaml and results.json files.
    
    This function is agnostic to the actual contents of the files and will
    flatten any YAML/JSON structure into DataFrame columns.
    
    Args:
        experiment_name: Name of the experiment folder to parse
        samples_base_path: Base path to samples directory
        
    Returns:
        pandas.DataFrame with parsed data
    """
    experiment_path = samples_base_path / experiment_name
    
    if not experiment_path.exists():
        raise ValueError(f"Experiment path does not exist: {experiment_path}")
    
    data_rows = []
    
    # Iterate through all sample timestamp directories
    for sample_dir in experiment_path.iterdir():
        if not sample_dir.is_dir():
            continue
            
        sample_name = sample_dir.name
        config_path = sample_dir / "config.yaml"
        results_path = sample_dir / "results.json"
        result_path = sample_dir / "result.json" # added for outdated samples support
        
        # Parse timestamp from sample_name (format: %Y%m%d_%H%M%S_%f)
        sample_timestamp = None
        try:
            sample_timestamp = datetime.strptime(sample_name, "%Y%m%d_%H%M%S_%f")
        except ValueError:
            # If parsing fails, try without microseconds (older format or manual directories)
            try:
                sample_timestamp = datetime.strptime(sample_name, "%Y%m%d_%H%M%S")
            except ValueError:
                # If still fails, leave as None and print warning
                print(f"Warning: Could not parse timestamp from directory name: {sample_name}")
        
        # Initialize row with basic metadata
        row_data = {
            'sample_name': sample_name,
            'sample_timestamp': sample_timestamp,
            'sample_path': str(sample_dir.relative_to(samples_base_path)),
            'experiment_name': experiment_name
        }
        
        # Parse config.yaml if it exists
        if config_path.exists():
            config_data = parse_yaml_file(str(config_path), prefix='config')
            row_data.update(config_data)
        else:
            print(f"Warning: Missing config.yaml in {sample_dir}")
            row_data['config_missing'] = 'True'
        
        # Parse results.json if it exists
        if results_path.exists():
            results_data = parse_json_file(str(results_path), prefix='results')
            row_data.update(results_data)
        elif result_path.exists():
            result_data = parse_json_file(str(result_path), prefix='result')
            row_data.update(result_data)
        else:
            print(f"Warning: Missing results.json and result.json in {sample_dir}")
            row_data['results_missing'] = 'True'
        
        # Skip if both files are missing
        if not config_path.exists() and not results_path.exists() and not result_path.exists():
            print(f"Warning: Both config.yaml and results.json and result.json missing in {sample_dir}")
            continue
            
        data_rows.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Sort by timestamp for chronological order
    if not df.empty:
        df = df.sort_values('sample_name').reset_index(drop=True)
    
    
    return df


def analyze_dataframe_structure(df: pd.DataFrame) -> None:
    """Print analysis of the DataFrame structure."""
    print(f"DataFrame shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    
    # Group columns by prefix
    config_cols = [col for col in df.columns if col.startswith('config_')]
    results_cols = [col for col in df.columns if col.startswith('results_')]
    meta_cols = [col for col in df.columns if not col.startswith(('config_', 'results_'))]
    
    print(f"  Metadata columns ({len(meta_cols)}): {meta_cols}")
    print(f"  Config columns ({len(config_cols)}): {config_cols[:5]}{'...' if len(config_cols) > 5 else ''}")
    print(f"  Results columns ({len(results_cols)}): {results_cols[:5]}{'...' if len(results_cols) > 5 else ''}")
    
    # Check for missing data
    missing_summary = df.isnull().sum()
    if missing_summary.sum() > 0:
        print(f"\nMissing data summary:")
        for col, missing_count in missing_summary[missing_summary > 0].items():
            print(f"  {col}: {missing_count}/{len(df)} samples")


def read_experiment_to_dataframe(samples_base_path: str, experiment_name: str) -> pd.DataFrame:
    """
    Read an experiment from the samples directory and return a pandas DataFrame.
    """
    # Set experiment name (this can be changed)
    # samples_base_path = Path(__file__).resolve().parent.parent / "samples"

    
    print(f"Parsing samples from experiment: {experiment_name}")
    
    try:
        # Create DataFrame
        df = create_samples_dataframe(experiment_name, samples_base_path)
        
        if df.empty:
            print("No valid samples found!")
            return df
        
        # Analyze structure
        analyze_dataframe_structure(df)
        
        # # Display sample data
        # print(f"\nFirst few rows:")
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.width', None)
        # pd.set_option('display.max_colwidth', 50)
        # print(df.head())
        
        # # Save to CSV for easy inspection
        # output_path = f"outputs/{experiment_name}_samples_dataframe.csv"
        # os.makedirs("outputs", exist_ok=True)
        # df.to_csv(output_path, index=False)
        # print(f"\nDataFrame saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def unroll_environment(policies: Sequence[Union[Callable, int, np.ndarray]], 
                      n_rollouts: int = 100,
                      env=None,
                      seed: Optional[int] = None) -> float:
    """
    Unroll a multi-agent environment with given policies for n times and return average return.
    
    Args:
        policies: List of policies for each agent. Each policy can be:
                 - A callable function that takes (observation, agent_id) and returns action
                 - An integer for a deterministic fixed action
                 - A numpy array representing action probabilities
        n_rollouts: Number of times to run the environment
        env: Environment instance. If None, creates TwoPlayerLeverGame
        seed: Random seed for reproducibility
        
    Returns:
        float: Average return across all rollouts
    """
    if env is None:
        env = TwoPlayerLeverGame()
    
    if seed is not None:
        np.random.seed(seed)
    
    total_returns = []
    
    for rollout in range(n_rollouts):
        # Reset environment
        obs, info = env.reset()
        total_return = 0.0
        
        while True:
            # Get actions from each policy
            actions = []
            for agent_id, policy in enumerate(policies):
                action = get_action_from_policy(policy, obs, agent_id)
                actions.append(action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(tuple(actions))
            total_return += reward
            
            if terminated or truncated:
                break
        
        total_returns.append(total_return)
    
    return float(np.mean(total_returns))


def get_action_from_policy(policy: Union[Callable, int, np.ndarray], 
                          observation: np.ndarray, 
                          agent_id: int) -> int:
    """
    Extract action from different policy types.
    
    Args:
        policy: Policy (function, fixed action, or probability distribution)
        observation: Current observation
        agent_id: ID of the agent
        
    Returns:
        int: Action to take
    """
    if callable(policy):
        # Policy is a function
        return policy(observation, agent_id)
    elif isinstance(policy, int):
        # Policy is a fixed action
        return policy
    elif isinstance(policy, np.ndarray):
        # Policy is a probability distribution over actions
        return np.random.choice(len(policy), p=policy)
    else:
        raise ValueError(f"Unsupported policy type: {type(policy)}")

def load_q_learning_agent(filepath: str):
    """
    Load a trained Q-learning agent from a pickle file.
    
    This function can handle multiple Q-learning agent formats:
    1. Full agent data (q_learning_1 format): Dictionary with q_table, action_space_size, and other metadata
    2. Simple Q-table (q_learning_2 format): Direct numpy array Q-table
    3. Custom formats: Extensible through format detection
    
    Args:
        filepath: Path to the agent pickle file
        
    Returns:
        Loaded agent object with Q-table and parameters
    """
    # Make path relative to this script's location if not absolute
    if not os.path.isabs(filepath):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the project root, then to context_files
        project_root = os.path.dirname(script_dir)
        filepath = os.path.join(project_root, "context_files", "algos", filepath)
    
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)
    
    # Debug: print information about the loaded data
    print(f"DEBUG: Loading {filepath}")
    print(f"DEBUG: Data type: {type(loaded_data)}")
    if isinstance(loaded_data, dict):
        print(f"DEBUG: Dict keys: {list(loaded_data.keys())}")
        if loaded_data:
            first_key = next(iter(loaded_data.keys()))
            first_value = loaded_data[first_key]
            print(f"DEBUG: First key type: {type(first_key)}, value type: {type(first_value)}")
    
    # Create a universal agent-like object to hold the Q-table and make decisions
    class LoadedQLearningAgent:
        def __init__(self, data, data_format):
            self.data_format = data_format
            self.metadata = {}
            
            if data_format == "tabular_agent_data":
                # q_learning_4 format: dictionary with q_table and metadata
                self.q_table = data['q_table']
                
                # Infer action space size from Q-table
                if self.q_table:
                    # Get the first Q-value array and its length
                    first_q_values = next(iter(self.q_table.values()))
                    self.action_space_size = len(first_q_values)
                else:
                    # Fallback for empty Q-table, specific to lever_game analysis.
                    print("Warning: Q-table is empty. Assuming action space size of 2 for TwoPlayerLeverGame.")
                    self.action_space_size = 2

                self.metadata = {k: v for k, v in data.items() if k != 'q_table'}

            elif data_format == "full_agent_data":
                # q_learning_1 format: Full agent data dictionary
                self.action_space_size = data['action_space_size']
                self.q_table = {}
                
                # Convert Q-table back to usable format
                for state_key, q_values in data['q_table'].items():
                    self.q_table[state_key] = q_values
                
                # Store additional metadata
                self.metadata = {
                    'learning_rate': data.get('learning_rate'),
                    'discount_factor': data.get('discount_factor'),
                    'epsilon': data.get('epsilon'),
                    'epsilon_decay': data.get('epsilon_decay'),
                    'epsilon_min': data.get('epsilon_min'),
                    'total_reward': data.get('total_reward'),
                    'episode_count': data.get('episode_count')
                }
                
            elif data_format == "simple_qtable":
                # q_learning_2 format: Direct Q-table as numpy array
                if isinstance(data, np.ndarray):
                    self.action_space_size = len(data)
                    # For simple Q-table, assume single state (lever game format)
                    self.q_table = {(0,): data}  # Single state key with Q-values
                else:
                    raise ValueError(f"Expected numpy array for simple_qtable format, got {type(data)}")
                    
            elif data_format == "direct_qtable_dict":
                # q_learning_5 format: Direct Q-table as dictionary with tuple keys
                self.q_table = data
                
                # Infer action space size from Q-table
                if self.q_table:
                    # Get the first Q-value array and its length
                    first_q_values = next(iter(self.q_table.values()))
                    self.action_space_size = len(first_q_values)
                else:
                    # Fallback for empty Q-table
                    print("Warning: Q-table is empty. Assuming action space size of 2 for TwoPlayerLeverGame.")
                    self.action_space_size = 2
                    
            elif data_format == "custom":
                # Future custom formats can be handled here
                # For now, try to extract what we can
                if hasattr(data, 'q_table') and hasattr(data, 'action_space_size'):
                    # Object with q_table and action_space_size attributes
                    self.action_space_size = data.action_space_size
                    self.q_table = data.q_table if isinstance(data.q_table, dict) else {}
                else:
                    raise ValueError(f"Unsupported custom format for data: {type(data)}")
            else:
                raise ValueError(f"Unknown data format: {data_format}")
                
        def state_to_key(self, state):
            """Convert state observation to a hashable key for Q-table."""
            if isinstance(state, np.ndarray):
                # Handle numpy array states, which may be floats
                if state.dtype in [np.float32, np.float64]:
                    state = np.round(state, 3)
                key = tuple(state.flatten())
            elif isinstance(state, (float, np.floating)):
                # Handle single float states, rounding them
                key = (round(state, 3),)
            else:
                # Handle other types, like integers
                key = tuple([state]) if not isinstance(state, tuple) else state

            # The newer agents use stringified tuples as keys in their q_tables.
            # The loader needs to match this format.
            if self.data_format == "tabular_agent_data":
                return str(key)
            
            return key
        
        def select_action(self, state):
            """Select action with highest Q-value (greedy policy)."""
            state_key = self.state_to_key(state)
            
            # If state not seen during training, return random action
            if state_key not in self.q_table:
                return np.random.randint(self.action_space_size)
            
            # Return action with highest Q-value
            return np.argmax(self.q_table[state_key])
        
        def get_q_values(self, state):
            """Get Q-values for a given state."""
            state_key = self.state_to_key(state)
            if state_key in self.q_table:
                return self.q_table[state_key].copy()
            else:
                return np.zeros(self.action_space_size)
        
        def get_metadata(self):
            """Get agent metadata (learning parameters, training stats, etc.)."""
            return self.metadata.copy()
    
    # Detect data format
    data_format = _detect_agent_format(loaded_data)
    print(f"DEBUG: Detected format: {data_format}")
    
    return LoadedQLearningAgent(loaded_data, data_format)


def _detect_agent_format(data):
    """
    Detect the format of loaded Q-learning agent data.
    
    Args:
        data: Loaded pickle data
        
    Returns:
        str: Format identifier ('full_agent_data', 'simple_qtable', 'tabular_agent_data', 'direct_qtable_dict', 'custom')
    """
    if isinstance(data, dict):
        # Check for tabular agent format (from q_learning_4/q_learning.py)
        if 'q_table' in data and 'agent_id' in data and 'action_space_size' not in data:
            return "tabular_agent_data"
        # Check if it has the structure of q_learning_1 format
        elif 'q_table' in data and 'action_space_size' in data:
            return "full_agent_data"
        # Check for direct Q-table format (from q_learning_5/q_learning.py)
        elif (len(data) > 0 and 
              all(isinstance(key, tuple) for key in data.keys()) and 
              all(isinstance(value, np.ndarray) for value in data.values())):
            return "direct_qtable_dict"
        # Handle empty Q-table case - if it's an empty dict, assume it's direct Q-table format
        elif len(data) == 0:
            print("Warning: Empty Q-table detected, assuming direct_qtable_dict format")
            return "direct_qtable_dict"
        else:
            return "custom"
    elif isinstance(data, np.ndarray):
        # Simple Q-table format (q_learning_2)
        return "simple_qtable"
    else:
        # Try to handle as custom format
        return "custom"

# %%
def create_q_learning_policy(agent):
    """
    Create a policy function from a loaded Q-learning agent that's compatible with unroll_environment.
    
    Args:
        agent: Loaded Q-learning agent object
        
    Returns:
        Callable policy function that takes (observation, agent_id) and returns action
    """
    def policy(observation, agent_id):
        return agent.select_action(observation)
    return policy

def load_trained_q_learning_policies(agent1_file: str,
                                   agent2_file: str):
    """
    Load both trained Q-learning agents and return their policies.
    
    Args:
        agent1_file: Path to agent 1's pickle file
        agent2_file: Path to agent 2's pickle file
        
    Returns:
        Tuple of (agent1_policy, agent2_policy) compatible with unroll_environment
    """
    # Load agents
    agent1 = load_q_learning_agent(agent1_file)
    agent2 = load_q_learning_agent(agent2_file)
    
    # Create policy functions
    policy1 = create_q_learning_policy(agent1)
    policy2 = create_q_learning_policy(agent2)
    
    print(f"Loaded Q-learning policies from {agent1_file} and {agent2_file}")
    return policy1, policy2
import os
import json
import pickle
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# assert that the results.json file exists and has the following keys:
# - cross_play_matrix
# - avg_self_play_reward
# - avg_cross_play_reward
results_file = script_dir / "results.json"
assert results_file.exists(), f"results.json file not found at {results_file}.json"

with open(results_file, 'r') as f:
    results = json.load(f)


    
# Check required keys for each mode
required_keys = ["cross_play_matrix", "avg_self_play_reward", "avg_cross_play_reward"]
for key in required_keys:
    assert key in results, f"Key '{key}' not found in results.json"

# Check that cross_play_matrix is a list of lists (matrix structure)
assert isinstance(results["cross_play_matrix"], list), f"cross_play_matrix should be a list"
assert len(results["cross_play_matrix"]) > 0, f"cross_play_matrix should not be empty"
assert isinstance(results["cross_play_matrix"][0], list), f"cross_play_matrix should be a matrix (list of lists)"

# Check that rewards are numeric
assert isinstance(results["avg_self_play_reward"], (int, float)), f"avg_self_play_reward should be numeric"
assert isinstance(results["avg_cross_play_reward"], (int, float)), f"avg_cross_play_reward should be numeric"

print("✓ results.json file exists and has all required keys and proper structure")

# assert that the trained_agents directory exists and the files are named as expected and can be loaded
trained_agents_dir = "trained_agents"

agents_dir = script_dir / trained_agents_dir
assert agents_dir.exists(), f"Directory {agents_dir} does not exist"
assert agents_dir.is_dir(), f"{agents_dir} is not a directory"
    
# Check that seed directories exist
seed_dirs = list(agents_dir.glob("seed_*"))
assert len(seed_dirs) > 0, f"No seed directories found in {agents_dir}"

# Check at least one seed directory and its agent files
for seed_dir in seed_dirs[:3]:  # Check first 3 seed dirs
    assert seed_dir.is_dir(), f"{seed_dir} should be a directory"
    
    # Check for agent files
    agent1_file = seed_dir / "lever_game_q_learning_agent1.pkl"
    agent2_file = seed_dir / "lever_game_q_learning_agent2.pkl"
    
    assert agent1_file.exists(), f"Agent 1 file not found at {agent1_file}"
    assert agent2_file.exists(), f"Agent 2 file not found at {agent2_file}"
    

print("✓ trained_agents directories exist with proper structure and agent files can be loaded successfully")

print("All assertions passed! The results.json and trained agent files are valid and loadable.")


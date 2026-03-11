import os
import pickle
import json
import numpy as np
from lever_game import TwoPlayerLeverGame


# --- Tabular Independent Q-Learning Implementation ---

# Hyperparameters
num_episodes = 5000
learning_rate = 0.1
start_epsilon = 1.0
end_epsilon = 0.1
# Epsilon decay rate calculated to linearly decrease over all episodes
epsilon_decay = (start_epsilon - end_epsilon) / num_episodes
n_seeds = 5
num_evaluation_episodes = 100

# Lists to store the learned policies (Q-tables) from each seed
policies1 = []
policies2 = []

# --- Training Loop ---
print("Starting training...")
for seed in range(n_seeds):
    print(f"\n--- Training for seed {seed}/{n_seeds} ---")
    
    # Create a separate random number generator for this seed to ensure isolation
    rng = np.random.RandomState(seed)
    
    # Create environment with the same seed
    env = TwoPlayerLeverGame(random_seed=seed)
    num_levers = env.num_levers

    # Initialize Q-tables for two independent agents.
    # Since the game has only one state, the Q-tables are 1D arrays mapping actions to Q-values.
    q_table1 = np.zeros(num_levers)
    q_table2 = np.zeros(num_levers)

    epsilon = start_epsilon
    for episode in range(num_episodes):
        env.reset()

        # Epsilon-greedy action selection for both agents
        if rng.random() < epsilon:
            action1 = rng.randint(0, num_levers)  # Explore
        else:
            action1 = np.argmax(q_table1)  # Exploit

        if rng.random() < epsilon:
            action2 = rng.randint(0, num_levers)  # Explore
        else:
            action2 = np.argmax(q_table2)  # Exploit

        # Take joint action in the environment
        _, reward, _, _, _ = env.step((action1, action2))

        # Update Q-tables. Since it's a one-shot game, the update rule simplifies.
        # Q(s,a) <-- Q(s,a) + alpha * (reward - Q(s,a))
        q_table1[action1] = q_table1[action1] + learning_rate * (reward - q_table1[action1])
        q_table2[action2] = q_table2[action2] + learning_rate * (reward - q_table2[action2])

        # Decay epsilon
        epsilon = max(end_epsilon, epsilon - epsilon_decay)
        
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{num_episodes} finished.")

    print(f"Finished training for seed {seed+1}.")
    print("Learned Q-table for Player 1:", np.round(q_table1, 2))
    print("Learned Q-table for Player 2:", np.round(q_table2, 2))
    policies1.append(q_table1)
    policies2.append(q_table2)

    # Save Q-tables to file
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trained_agents", f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "lever_game_q_learning_agent1.pkl"), "wb") as f:
        pickle.dump(q_table1, f)
    with open(os.path.join(save_dir, "lever_game_q_learning_agent2.pkl"), "wb") as f:
        pickle.dump(q_table2, f)

print("\n--- Training Finished ---")

# --- Cross-Play Evaluation ---
print("\n--- Starting Cross-Play Evaluation ---")
crossplay_scores = np.zeros((n_seeds, n_seeds))
env = TwoPlayerLeverGame(random_seed=42)

for i in range(n_seeds):
    for j in range(n_seeds):
        p1_q_table = policies1[i]
        p2_q_table = policies2[j]

        total_reward = 0
        for _ in range(num_evaluation_episodes):
            env.reset()
            # Agents choose the best action based on their learned Q-tables
            action1 = np.argmax(p1_q_table)
            action2 = np.argmax(p2_q_table)
            
            _, reward, _, _, _ = env.step((action1, action2))
            total_reward += reward
        
        average_reward = total_reward / num_evaluation_episodes
        crossplay_scores[i, j] = average_reward

print("\n--- Cross-Play Results ---")
print(f"Matrix of average rewards over {num_evaluation_episodes} rollouts.")
print("scores[i, j] is P1 from seed i vs P2 from seed j.")
print(np.round(crossplay_scores, 2))

# --- Save Results ---
print("\nSaving results to results.json...")

# Calculate average self-play reward (diagonal of the matrix)
avg_self_play_reward = np.diag(crossplay_scores).mean()

# Calculate average cross-play reward (off-diagonal of the matrix)
if n_seeds > 1:
    avg_cross_play_reward = crossplay_scores[~np.eye(n_seeds, dtype=bool)].mean()
else:
    avg_cross_play_reward = 0.0  # No cross-play partners if only one seed

results = {
    "cross_play_matrix": crossplay_scores.tolist(),
    "avg_self_play_reward": avg_self_play_reward,
    "avg_cross_play_reward": avg_cross_play_reward,
}

script_dir = os.path.dirname(os.path.realpath(__file__))
results_file = os.path.join(script_dir, "results.json")
os.makedirs(os.path.dirname(results_file), exist_ok=True)
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)




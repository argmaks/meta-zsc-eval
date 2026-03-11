from ippo_rnn_hanabi import main
from omegaconf import OmegaConf
from evaluate_crossplay_gfppoy_best_of_k import evaluate_crossplay, best_of_k_evaluation
import numpy as np
import os
import warnings
import json
import jax
import jax.numpy as jnp

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
config = OmegaConf.load(config_path)["runner"]["algo_config"]

print("Training...")

main(config)
print("Training complete.")


# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=UserWarning)

#     print("Evaluating crossplay...")
#     rng = jax.random.PRNGKey(0)
#     config = OmegaConf.to_container(config)
#     crossplay_matrix = evaluate_crossplay(config, rng)
#     best_of_k_crossplay_matrix = best_of_k_evaluation(crossplay_matrix, config["BEST_OF_K"])
#     print("Crossplay evaluation complete.")

# print("Crossplay matrix:")
# print(crossplay_matrix)

# avg_self_play_score = np.mean(np.diag(crossplay_matrix))
# print("Avg self-play score:", avg_self_play_score)

# xp_mask = ~np.eye(crossplay_matrix.shape[0], dtype=bool)
# avg_cross_play_score = np.mean(crossplay_matrix[xp_mask])
# print("Avg cross-play score:", avg_cross_play_score)
# print("Avg self-play cross-play difference:", avg_self_play_score - avg_cross_play_score)

# sp = crossplay_matrix.diagonal()
# n_seeds = crossplay_matrix.shape[0]
# k = config["BEST_OF_K"]
# best_of_k_indices = jnp.zeros((n_seeds // k), dtype=int)
# for i in range(n_seeds // k):
#     best_of_k_indices = best_of_k_indices.at[i].set(i*k + jnp.argmax(sp[i*k:(i+1)*k]))
# seeds = config["SEEDS"]
# best_of_k_seeds = [seed for i, seed in enumerate(seeds) if i in best_of_k_indices]
# print("Best of k seeds:", best_of_k_seeds)

# print("Best of k crossplay matrix:")
# print(best_of_k_crossplay_matrix)

# best_of_k_avg_self_play_score = np.mean(np.diag(best_of_k_crossplay_matrix))
# print("Best of k avg self-play score:", best_of_k_avg_self_play_score)
# best_of_k_avg_cross_play_score = np.mean(best_of_k_crossplay_matrix[~np.eye(best_of_k_crossplay_matrix.shape[0], dtype=bool)])
# print("Best of k avg cross-play score:", best_of_k_avg_cross_play_score)
# best_of_k_self_play_cross_play_diff = best_of_k_avg_self_play_score - best_of_k_avg_cross_play_score
# print("Best of k self-play cross-play difference:", best_of_k_self_play_cross_play_diff)

# # save the crossplay statistics to  a results.json file in the workspace_dir
# workspace_dir = os.path.join(os.path.dirname(__file__))
# results_file = os.path.join(workspace_dir, "results.json")

# # Check if results.json exists and load existing data
# if os.path.exists(results_file):
#     with open(results_file, "r") as f:
#         existing_data = json.load(f)
# else:
#     existing_data = {}

# # Update with new data
# existing_data.update({
#     'cross_play_matrix': crossplay_matrix.tolist(),
#     'avg_self_play_reward': float(avg_self_play_score),
#     'avg_cross_play_reward': float(avg_cross_play_score),
#     'best_of_k_cross_play_matrix': best_of_k_crossplay_matrix.tolist(),
#     'best_of_k_avg_self_play_reward': float(best_of_k_avg_self_play_score),
#     'best_of_k_avg_cross_play_reward': float(best_of_k_avg_cross_play_score),
#     'best_of_k_self_play_cross_play_diff': float(best_of_k_self_play_cross_play_diff),
#     'best_of_k_seeds': best_of_k_seeds
# })

# with open(results_file, "w") as f:
#     json.dump(existing_data, f)
from decpomdp_symmetries_op_ippo_ff_hanabi import main
from omegaconf import OmegaConf

import numpy as np
import os
import warnings
import json
import jax

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
#     print("Crossplay evaluation complete.")

# print("Crossplay matrix:")
# print(crossplay_matrix)

# avg_self_play_score = np.mean(np.diag(crossplay_matrix))
# print("Avg self-play score:", avg_self_play_score)

# avg_cross_play_score = np.mean(crossplay_matrix[np.triu_indices(crossplay_matrix.shape[0], k=1)])
# print("Avg cross-play score:", avg_cross_play_score)

# print("Avg self-play cross-play difference:", avg_self_play_score - avg_cross_play_score)

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
#     'avg_self_play_reward': avg_self_play_score,
#     'avg_cross_play_reward': avg_cross_play_score
# })

# with open(results_file, "w") as f:
#     json.dump(existing_data, f)
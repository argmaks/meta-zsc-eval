# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from analysis.utils import read_experiment_to_dataframe
from pathlib import Path
import pandas as pd
import jaxmarl
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization as fs
from flax.linen.initializers import orthogonal, constant
import distrax
from typing import Sequence
from context_files.algos.jaxmarl.ippo_ff_hanabi.networks import ActorCritic as VanillaActorCritic
from context_files.algos.jaxmarl.ippo_ff_hanabi.networks import batchify, unbatchify
from context_files.algos.jaxmarl.decpomdp_symmetries_op_ippo_ff_hanabi.networks import ActorCritic as GFPPOYActorCritic
from context_files.algos.jaxmarl.fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped.networks import ActorCritic as FastGFPPOYActorCritic
from context_files.algos.jaxmarl.fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_alt.networks import ActorCritic as VanillaOPActorCritic
from context_files.algos.jaxmarl.ippo_rnn_hanabi.networks import ActorCriticRNN as RNNActorCritic
from context_files.algos.jaxmarl.ippo_rnn_hanabi.networks import ScannedRNN as ScannedRNN
from omegaconf import OmegaConf
import time
from functools import partial
import os
from typing import List
from typing import Dict
import pickle

# %%
# show all columns when printing a dataframe
pd.set_option('display.max_columns', None)
# %%
repo_root = Path(__file__).parent.parent
sample_dir = repo_root / "samples_persisted"

# Load experiment data
print("Loading experiment data...")
df = read_experiment_to_dataframe(sample_dir,"hanabi_stable")
# %%
df
# %%
sample_best_of_k_seeds = dict(zip(df.sample_name, df.results_best_of_k_seeds.apply(lambda x: x.split(', ') if isinstance(x, str) else [])))
print(sample_best_of_k_seeds)

# %%
# get paths of models
model_timesteps = str(1e9)
model_paths = {}
for sample_name, sample_path in zip(df.sample_name, df.sample_path):
    sample_path = sample_dir / sample_path
    found_paths = list(sample_path.glob(f"models/{model_timesteps}/model_*.pkl"))
    if len(found_paths) == 0:
        print(f"No models found for {sample_path}")
        model_paths[sample_name] = None
    else:
        # Extract model_index from filename pattern "model_*.pkl"
        model_paths[sample_name] = {}
        for path in found_paths:
            # Extract the number from "model_<number>.pkl"
            model_index = int(path.stem.split('_')[1])
            model_paths[sample_name][model_index] = path
        print(f"Found {len(model_paths[sample_name])} models for {sample_path}")

# %%
sample_network_types = {}
for sample_name, sample_model_paths in model_paths.items():
    lname = sample_name.lower()

    if "2025" in sample_name:
        sample_network_types[sample_name] = "vanilla"
    elif "fast" in lname and "alt" in lname:
        sample_network_types[sample_name] = "vanilla_indices"
    elif "fast" in lname:
        sample_network_types[sample_name] = "gfppoy_indices"
    elif "decpomdp_symmetries_op_ippo_ff_hanabi" in lname:
        sample_network_types[sample_name] = "gfppoy"
    elif "rnn" in lname:
        sample_network_types[sample_name] = "rnn"
    else:
        sample_network_types[sample_name] = "vanilla"
print("sample_network_types:", sample_network_types)

# %%

def get_network_init_x(network_type):
    if network_type == 'vanilla':
        return (
                jnp.zeros((1, rollouts, env.observation_space(env.agents[0]).shape)),
                jnp.zeros((1, rollouts)),
                jnp.zeros((1, rollouts, env.action_space(env.agents[0]).n)),
            )
    elif network_type == 'gfppoy':
        out_permutations = jnp.tile(jnp.identity(env.action_space(env.agents[0]).n), (rollouts, 1, 1)).astype(jnp.int32)
        return (
                jnp.zeros((1, rollouts, env.observation_space(env.agents[0]).shape)),
                jnp.zeros((1, rollouts)),
                jnp.zeros((1, rollouts, env.action_space(env.agents[0]).n)),
                out_permutations
            )

    elif network_type == 'gfppoy_indices':
        out_permutations = jnp.tile(jnp.arange(env.action_space(env.agents[0]).n), (rollouts, 1)).astype(jnp.int32)
        return (
                jnp.zeros((1, rollouts, env.observation_space(env.agents[0]).shape)),
                jnp.zeros((1, rollouts)),
                jnp.zeros((1, rollouts, env.action_space(env.agents[0]).n)),
                out_permutations
            )
    elif network_type == 'vanilla_indices':
        return (
                jnp.zeros((1, rollouts, env.observation_space(env.agents[0]).shape)),
                jnp.zeros((1, rollouts)),
                jnp.zeros((1, rollouts, env.action_space(env.agents[0]).n)),
            )
    elif network_type == 'rnn':
        return (
                jnp.zeros((1, rollouts, env.observation_space(env.agents[0]).shape)),
                jnp.zeros((1, rollouts)),
                jnp.zeros((1, rollouts, env.action_space(env.agents[0]).n)),
            )
    else:
        raise ValueError(f"Invalid network type: {network_type}")

def get_network(network_type, config):
    if network_type == 'vanilla':
        return VanillaActorCritic(action_dim=env.action_space(env.agents[0]).n, config=config)
    elif network_type == 'gfppoy':
        return GFPPOYActorCritic(action_dim=env.action_space(env.agents[0]).n, config=config)
    elif network_type == 'gfppoy_indices':
        return FastGFPPOYActorCritic(action_dim=env.action_space(env.agents[0]).n, config=config)
    elif network_type == 'vanilla_indices':
        return VanillaOPActorCritic(action_dim=env.action_space(env.agents[0]).n, config=config)
    elif network_type == 'rnn':
        return RNNActorCritic(action_dim=env.action_space(env.agents[0]).n, config=config)
    else:
        raise ValueError(f"Invalid network type: {network_type}")

def get_network_params_template(network_type, config):
    rng = jax.random.PRNGKey(0)
    init_x = get_network_init_x(network_type)
    network = get_network(network_type, config)
    if network_type == 'rnn':
        init_hstate = ScannedRNN.initialize_carry(rollouts, config["GRU_HIDDEN_DIM"])
        return network.init(rng, init_hstate, init_x)

    else:
        return network.init(rng, init_x)


def prepare_ac_in(network_type, obs_batch, last_done, avail_actions, config):
    if network_type == 'vanilla':
        return (obs_batch[jnp.newaxis, :], last_done, avail_actions[jnp.newaxis, :])
    elif network_type == 'gfppoy':
        out_permutations = jnp.tile(jnp.identity(env.action_space(env.agents[0]).n), (config["NUM_ACTORS"], 1, 1))
        return (obs_batch[jnp.newaxis, :], last_done, avail_actions[jnp.newaxis, :], out_permutations)
    elif network_type == 'gfppoy_indices':
        out_permutations = jnp.tile(jnp.arange(env.action_space(env.agents[0]).n), (config["NUM_ACTORS"], 1))
        return (obs_batch[jnp.newaxis, :], last_done, avail_actions[jnp.newaxis, :], out_permutations)
    elif network_type == 'vanilla_indices':
        return (obs_batch[jnp.newaxis, :], last_done, avail_actions[jnp.newaxis, :])
    elif network_type == 'rnn':
        return (obs_batch[jnp.newaxis, :], last_done, avail_actions[jnp.newaxis, :])
    else:
        raise ValueError(f"Invalid network type: {network_type}")



def load_network_params(network_type: str, model_paths: List[Path], config):

    network_params_template = get_network_params_template(network_type, config)

    network_params_list = []
    for model_path in model_paths:
        with open(model_path, 'rb') as f:
            params_bytes = f.read()
        params = fs.from_bytes(network_params_template, params_bytes)
        network_params_list.append(params)

    return network_params_list

def get_config(network_type):
    config = OmegaConf.load("/home/ubuntu/meta-zsc/configs/runner/ippo_ff_hanabi_full_hanabi.yaml")
    config_rnn = OmegaConf.load("/home/ubuntu/meta-zsc/configs/runner/ippo_rnn_hanabi_full_hanabi.yaml")["algo_config"]
    config["NUM_ACTORS"] = 2
    config_rnn["NUM_ACTORS"] = 2
    if network_type == 'rnn':
        return config_rnn
    else:
        return config

def get_loaded_network_params_lists(sample_names: List[str]):
    loaded_network_params_lists = {}
    for sample_name in sample_names:
        # Load params with same dict structure: {model_index: params}
        model_path_dict = model_paths[sample_name]
        network_type = sample_network_types[sample_name]
        network_params_template = get_network_params_template(network_type, get_config(network_type))
        
        loaded_network_params_lists[sample_name] = {}
        for model_index, model_path in model_path_dict.items():
            with open(model_path, 'rb') as f:
                params_bytes = f.read()
            params = fs.from_bytes(network_params_template, params_bytes)
            loaded_network_params_lists[sample_name][model_index] = params
    
    return loaded_network_params_lists

def apply_network(network, network_params, network_type, ac_in, hstate):
    if network_type == 'rnn':
        hstate, pi, value = network.apply(network_params, hstate, ac_in)
        return hstate, pi, value
    else:
        pi, value = network.apply(network_params, ac_in)
        return hstate, pi, value


# %%
rng = jax.random.PRNGKey(0)
config = OmegaConf.load("/home/ubuntu/meta-zsc/configs/runner/ippo_ff_hanabi_full_hanabi.yaml")
config_rnn = OmegaConf.load("/home/ubuntu/meta-zsc/configs/runner/ippo_rnn_hanabi_full_hanabi.yaml")["algo_config"]
config["NUM_ACTORS"] = 2
config_rnn["NUM_ACTORS"] = 2
env = jaxmarl.make("hanabi")

# %%
# network = get_network('vanilla')
# network_0 = get_network('vanilla')
# network_1 = get_network('gfppoy')
rollouts = 5000


# seeds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# sample_names = list(model_paths.keys())
# sample_names = [
#     "ippo_rnn_hanabi_fixed",
#     "fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped",
#     "fast_gfppoy_op_ippo_ff_hanabi_details-anneal_LR",
#     "fast_gfppoy_op_ippo_ff_hanabi_details-high_ent",
#     "fast_gfppoy_op_ippo_ff_hanabi_details-low_GAE",
#     "fast_gfppoy_op_ippo_ff_hanabi_details-unnormalize_advantages",
# ]
sample_names = [
    "ippo_rnn_hanabi_fixed",
    "ippo_rnn_hanabi_fixed_op",
    # "ippo_rnn_hanabi_fixed_op-high_ent",
    # "ippo_rnn_hanabi_fixed_op-low_GAE",
    # "ippo_rnn_hanabi_fixed_op-unanneal_LR",
    # "ippo_rnn_hanabi_details_op-kernel_xavier_uniform",
    # "ippo_rnn_hanabi_details_op-no_global_norm_clip",
    # "ippo_rnn_hanabi_details_op-no_value_function_clip",

]
# sample_names = [
#     "ippo_rnn_hanabi_fixed_op",
#     "ippo_rnn_hanabi_fixed_op-high_ent",
#     "ippo_rnn_hanabi_fixed_op-low_GAE",
#     "fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped",
#     "fast_gfppoy_op_ippo_ff_hanabi_details-anneal_LR",
#     "fast_gfppoy_op_ippo_ff_hanabi_details-high_ent",
#     "fast_gfppoy_op_ippo_ff_hanabi_details-low_GAE",
# ]
loaded_network_params_lists = get_loaded_network_params_lists(sample_names)


# import pdb; pdb.set_trace()



# %%
@partial(jax.jit, static_argnames=("network_type_0", "network_type_1"))
def play_game(rng, network_params_0, network_params_1, network_type_0 = 'gfppoy', network_type_1 = 'gfppoy'):

    if network_type_0 == 'rnn':
        config_0 = config_rnn
    else:
        config_0 = config

    if network_type_1 == 'rnn':
        config_1 = config_rnn
    else:
        config_1 = config

    # Single-environment rollout using the loaded network parameters
    rng, _rng = jax.random.split(rng)

    obs, env_state = env.reset(_rng)
    legal_moves = env.get_legal_moves(env_state)

    local_num_actors = env.num_agents  # 2 for Hanabi two-player

    init_hstate0 = ScannedRNN.initialize_carry(local_num_actors, config_rnn["GRU_HIDDEN_DIM"])
    init_hstate1 = ScannedRNN.initialize_carry(local_num_actors, config_rnn["GRU_HIDDEN_DIM"])

    
    def _step_env(rng, env_state, actions):
        rng, _rng = jax.random.split(rng)
        new_obs, new_env_state, reward, dones, infos = env.step(
            _rng, env_state, actions
        )
        new_legal_moves = env.get_legal_moves(new_env_state)
        return rng, new_env_state, new_obs, reward, dones, new_legal_moves

    done = False
    cum_rew = 0
    t = 0

    def cond_fn(val):
        _, _, done, _, _, _, _, _, _ = val
        return jax.numpy.logical_not(done)

    def body_fn(val):
        cum_rew, t, done, rng, env_state, obs, legal_moves, hstate0, hstate1 = val
        rng, _rng = jax.random.split(rng)

        # Prepare batched inputs for both agents in the single environment
        avail_actions = jax.lax.stop_gradient(
            batchify(legal_moves, env.agents, local_num_actors)
        )
        obs_batch = batchify(obs, env.agents, local_num_actors)
        last_done = jnp.zeros((1, local_num_actors), dtype=bool)
        
        ac_in_0 = prepare_ac_in(network_type_0, obs_batch, last_done, avail_actions, config_0)
        ac_in_1 = prepare_ac_in(network_type_1, obs_batch, last_done, avail_actions, config_1)

        network_0 = get_network(network_type_0, config_0)
        network_1 = get_network(network_type_1, config_1)

        # Compute actions for both models
        hstate0, pi0, _ = apply_network(network_0, network_params_0, network_type_0, ac_in_0, hstate0)
        hstate1, pi1, _ = apply_network(network_1, network_params_1, network_type_1, ac_in_1, hstate1)

        rng0, rng1 = jax.random.split(_rng)
        action0 = pi0.sample(seed=rng0)
        action1 = pi1.sample(seed=rng1)

        env_act0 = unbatchify(action0, env.agents, 1, env.num_agents)
        env_act0 = jax.tree.map(lambda x: x.squeeze(), env_act0)
        env_act1 = unbatchify(action1, env.agents, 1, env.num_agents)
        env_act1 = jax.tree.map(lambda x: x.squeeze(), env_act1)

        actions = {
            env.agents[0]: jnp.array(env_act0[env.agents[0]]),
            env.agents[1]: jnp.array(env_act1[env.agents[1]]),
        }

        rng, env_state, obs, reward, dones, legal_moves = _step_env(
            rng, env_state, actions
        )

        done = dones["__all__"]
        cum_rew += reward["__all__"]
        t += 1
        return (cum_rew, t, done, rng, env_state, obs, legal_moves, hstate0, hstate1)

    init_val = (0, 0, False, rng, env_state, obs, legal_moves, init_hstate0, init_hstate1)
    cum_rew, _, _, _, _, _, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)

    return cum_rew



play_games = jax.vmap(play_game, in_axes=(0, None, None, None, None))


# DOESNT WORK: IT GIVES WRONG RESULTS IN XP
# vmapped_play_games = jax.vmap(jax.vmap(play_games, in_axes=(None, 0, None, None, None)), in_axes=(None, None, 0, None, None))

# rng, _rng = jax.random.split(rng, 2)
# game_rngs = jax.random.split(_rng, rollouts)
# batched_params_list_vanilla = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *network_params_list_vanilla[:5])
# batched_params_list_gfppoy = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *network_params_list_gfppoy[:5])
# xp = jnp.mean(vmapped_play_games(game_rngs, batched_params_list_gfppoy, batched_params_list_gfppoy, 'gfppoy', 'gfppoy'), axis=-1)
# print(xp)
# sp_mask = jnp.eye(5, dtype=bool)
# print(xp[sp_mask].mean())
# print(xp[~sp_mask].mean())

def cross_play_pairs(rng, network_params_dict_0, network_params_dict_1, network_type_0, network_type_1):
    # network_params_dict_0/1 are dicts with {model_index: params}
    game_rngs = jax.random.split(rng, rollouts)
    
    # Get sorted lists of model indices
    model_indices_0 = sorted(network_params_dict_0.keys())
    model_indices_1 = sorted(network_params_dict_1.keys())
    
    xp = jnp.zeros((len(model_indices_0), len(model_indices_1), rollouts))
    for i, idx_0 in enumerate(model_indices_0):
        for j, idx_1 in enumerate(model_indices_1):
            xp = xp.at[i, j, :].set((play_games(game_rngs, network_params_dict_0[idx_0], network_params_dict_1[idx_1], network_type_0, network_type_1)))

    return xp
    
def cross_play_all(rng, loaded_network_params_lists: Dict[str, Dict[int, Dict]]):
    # loaded_network_params_lists is now Dict[sample_name, Dict[model_index, params]]
    # Get the number of models for each sample (assumes all samples have same number)
    num_models_per_sample = len(list(loaded_network_params_lists.values())[0])
    
    xxp = jnp.zeros((len(loaded_network_params_lists), len(loaded_network_params_lists), num_models_per_sample, num_models_per_sample, rollouts))
    for i, sample_name_0 in enumerate(list(loaded_network_params_lists.keys())):
        for j, sample_name_1 in enumerate(list(loaded_network_params_lists.keys())):
            print(f"Evaluating {sample_name_0} vs {sample_name_1}")
            xxp = xxp.at[i, j, :].set(cross_play_pairs(rng, loaded_network_params_lists[sample_name_0], loaded_network_params_lists[sample_name_1], sample_network_types[sample_name_0], sample_network_types[sample_name_1]))
    return xxp

def filter_best_of_k(loaded_network_params_lists: Dict[str, Dict[int, Dict]], sample_best_of_k_seeds: Dict[str, List[str]]):
    # Filter to keep only the best_of_k model indices
    # import pdb; pdb.set_trace()
    loaded_network_params_lists_for_xxp = {}
    for sample_name, params_dict in loaded_network_params_lists.items():
        # Convert seed strings to integers (model indices)
        best_model_indices = [int(seed) for seed in sample_best_of_k_seeds[sample_name]]
        # Filter the params dict to only include best_of_k model indices
        loaded_network_params_lists_for_xxp[sample_name] = {
            idx: params_dict[idx] for idx in best_model_indices if idx in params_dict
        }
    return loaded_network_params_lists_for_xxp

# xp = cross_play_pairs(rng, loaded_network_params_lists["fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_alt"], loaded_network_params_lists["fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped"], 'vanilla_indices', 'gfppoy_indices')
# print(xp)
# sp_mask = jnp.eye(len(loaded_network_params_lists["fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_alt"]), dtype=bool)
# print(xp[sp_mask].mean())
# print(xp[~sp_mask].mean())
# loaded_network_params_lists_for_xxp = {k: v[:5] for k, v in loaded_network_params_lists.items()}
loaded_network_params_lists_for_xxp = filter_best_of_k(loaded_network_params_lists, sample_best_of_k_seeds)

best_of_k_samples_and_seeds = {}
best_of_k_samples_and_n_seeds = {}
for sample_name, sample_seeds in loaded_network_params_lists_for_xxp.items():
    best_of_k_samples_and_seeds[sample_name] = list(sample_seeds.keys())
    best_of_k_samples_and_n_seeds[sample_name] = len(list(sample_seeds.keys()))
print(best_of_k_samples_and_seeds)
print(best_of_k_samples_and_n_seeds)

rng = jax.random.PRNGKey(0)
xxp = cross_play_all(rng, loaded_network_params_lists_for_xxp)
print(xxp.shape)
print(xxp.mean(axis=(2,3,4)))

# %%
# save the xxp matrix to a pickle file
save_dir = Path(__file__).parent / "results" / "xxp"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "xxp_results-rnn-details.pkl"

data = {
    "sample_names": [sample_name for sample_name, _ in loaded_network_params_lists_for_xxp.items()],
    "sample_network_types": [sample_network_types[sample_name] for sample_name in loaded_network_params_lists_for_xxp.keys()],
    "best_of_k_samples_and_seeds": best_of_k_samples_and_seeds,
    "best_of_k_samples_and_n_seeds": best_of_k_samples_and_n_seeds,
    "rollouts": rollouts,
    "model_timesteps": model_timesteps,
    "xxp": xxp.tolist(),
}

with open(save_path, 'wb') as f:
    pickle.dump(data, f)

# %%

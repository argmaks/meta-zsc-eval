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
from context_files.algos.jaxmarl.ippo_ff_hanabi.networks import ActorCritic as IppoFfHanabiActorCritic
from context_files.algos.jaxmarl.ippo_ff_hanabi.networks import batchify, unbatchify
from context_files.algos.jaxmarl.decpomdp_symmetries_op_ippo_ff_hanabi.networks import ActorCritic as DecPomdpSymmetriesOpIppoFfHanabiActorCritic
from context_files.algos.jaxmarl.fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped.networks import ActorCritic as FastDecPomdpSymmetriesOpIppoFfHanabiPmappedActorCritic
from context_files.algos.jaxmarl.fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_alt.networks import ActorCritic as FastDecPomdpSymmetriesOpIppoFfHanabiPmappedActorCriticAlt
from omegaconf import OmegaConf
import time
from functools import partial
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
    model_paths[sample_name] = list(sample_path.glob(f"models/{model_timesteps}/model_*.pkl"))
    if len(model_paths[sample_name]) == 0:
        print(f"No models found for {sample_path}")
        model_paths[sample_name] = None
    else:
        print(f"Found {len(model_paths[sample_name])} models for {sample_path}")

# %%
config = OmegaConf.load("/home/ubuntu/meta-zsc/configs/runner/ippo_ff_hanabi_full_hanabi.yaml")
env = jaxmarl.make("hanabi")
ippo_ff_hanabi_network = IppoFfHanabiActorCritic(action_dim=env.action_space(env.agents[0]).n, config=config)
decpomdp_symmetries_op_ippo_ff_hanabi_network = DecPomdpSymmetriesOpIppoFfHanabiActorCritic(action_dim=env.action_space(env.agents[0]).n, config=config)
fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_network = FastDecPomdpSymmetriesOpIppoFfHanabiPmappedActorCritic(action_dim=env.action_space(env.agents[0]).n, config=config)
fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_alt_network = FastDecPomdpSymmetriesOpIppoFfHanabiPmappedActorCriticAlt(action_dim=env.action_space(env.agents[0]).n, config=config)

rng = jax.random.PRNGKey(0)

vanilla_init_x = (
    jnp.zeros((1, 1, env.observation_space(env.agents[0]).shape)),
    jnp.zeros((1, 1)),
    jnp.zeros((1, 1, env.action_space(env.agents[0]).n))
)
vanilla_alt_init_x = (
    jnp.zeros((1, 1, env.observation_space(env.agents[0]).shape)),
    jnp.zeros((1, 1)),
    jnp.zeros((1, 1, env.action_space(env.agents[0]).n))
)
permutations_init_x = (
    jnp.zeros((1, 5000, env.observation_space(env.agents[0]).shape)),
    jnp.zeros((1, 5000)),
    jnp.zeros((1, 5000, env.action_space(env.agents[0]).n)),
    jnp.tile(jnp.identity(env.action_space(env.agents[0]).n), (5000, 1, 1))
)
indices_permutation_init_x = (
    jnp.zeros((1, 1, env.observation_space(env.agents[0]).shape)),
    jnp.zeros((1, 1)),
    jnp.zeros((1, 1, env.action_space(env.agents[0]).n)),
    jnp.tile(jnp.arange(env.action_space(env.agents[0]).n), (1, 1))
)

vanilla_network_params_template = ippo_ff_hanabi_network.init(rng, vanilla_init_x)
decpomdp_symmetries_op_ippo_ff_hanabi_network_params_template = decpomdp_symmetries_op_ippo_ff_hanabi_network.init(rng, permutations_init_x)
fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_network_params_template = fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_network.init(rng, indices_permutation_init_x)
fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_alt_network_params_template = fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_alt_network.init(rng, vanilla_alt_init_x)
# %%
sample_network_types = {}
for sample_name, sample_model_paths in model_paths.items():
    lname = sample_name.lower()

    if "2025" in sample_name:
        sample_network_types[sample_name] = "vanilla"
    elif "fast" in lname and "alt" in lname:
        sample_network_types[sample_name] = "alt_indices"
    elif "fast" in lname:
        sample_network_types[sample_name] = "gfppoy_indices"
    elif "decpomdp_symmetries_op_ippo_ff_hanabi" in lname:
        sample_network_types[sample_name] = "gfppoy"
    elif "rnn" in lname:
        sample_network_types[sample_name] = "vanilla_rnn"
    elif "details" in lname:
        sample_network_types[sample_name] = "gfppoy_indices"
    else:
        sample_network_types[sample_name] = "vanilla"

print("sample_network_types:", sample_network_types)






# TODO: FIX!
# %%
loaded_params = {}
for sample_name, sample_model_paths in model_paths.items():
    sample_network_type = sample_network_types[sample_name]
    if sample_model_paths is None:
        print(f"Skipping {sample_name} due to no models found")
        continue
    loaded_params[sample_name] = {}
    for model_path in sample_model_paths:
        with open(model_path, 'rb') as f:
            params_bytes = f.read()
        try:
            if sample_network_type == 'vanilla':
                params = fs.from_bytes(vanilla_network_params_template, params_bytes)
            
            elif sample_network_type == 'gfppoy':
                params = fs.from_bytes(decpomdp_symmetries_op_ippo_ff_hanabi_network_params_template, params_bytes)
            elif sample_network_type == 'gfppoy_indices':
                params = fs.from_bytes(fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_network_params_template, params_bytes)
            elif sample_network_type == 'alt_indices':
                params = fs.from_bytes(fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_alt_network_params_template, params_bytes)
            else:
                params = None
            seed_id = str(model_path).split("/")[-1].split("_")[1].split(".")[0]
            loaded_params[sample_name][seed_id] = params

        except Exception as e:
            print(f"Failed to load model {model_path} for sample {sample_name}: {e}")



loaded_types = sample_network_types


# %%
# now we can evaluate the performance of the models

# %%
@partial(jax.jit, static_argnums=(3, 4))
def play_game(rng, network_params_0, network_params_1, network_type_0 = 'vanilla', network_type_1 = 'vanilla'):

    # Single-environment rollout using the loaded network parameters
    rng, _rng = jax.random.split(rng)

    obs, env_state = env.reset(_rng)
    legal_moves = env.get_legal_moves(env_state)

    local_num_actors = env.num_agents  # 2 for Hanabi two-player

    
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
        _, _, done, _, _, _, _ = val
        return jax.numpy.logical_not(done)

    def body_fn(val):
        cum_rew, t, done, rng, env_state, obs, legal_moves = val
        rng, _rng = jax.random.split(rng)

        # Prepare batched inputs for both agents in the single environment
        avail_actions = jax.lax.stop_gradient(
            batchify(legal_moves, env.agents, local_num_actors)
        )
        obs_batch = batchify(obs, env.agents, local_num_actors)
        last_done = jnp.zeros((1, local_num_actors), dtype=bool)

        if network_type_0 == 'vanilla':
            ac_in_0 = (
                obs_batch[jnp.newaxis, :],
                last_done,
                avail_actions[jnp.newaxis, :],
            )
            network_0 = ippo_ff_hanabi_network
        elif network_type_0 == 'alt_indices':
            ac_in_0 = (
                obs_batch[jnp.newaxis, :],
                last_done,
                avail_actions[jnp.newaxis, :],
            )
            network_0 = fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_alt_network
        elif network_type_0 == 'gfppoy':
            out_permutations = jnp.tile(jnp.identity(env.action_space(env.agents[0]).n), (2, 1, 1)) 
            ac_in_0 = (
                obs_batch[jnp.newaxis, :],
                last_done,
                avail_actions[jnp.newaxis, :],
                out_permutations
            )
            network_0 = decpomdp_symmetries_op_ippo_ff_hanabi_network
        elif network_type_0 == 'gfppoy_indices':
            out_permutations = jnp.tile(jnp.arange(env.action_space(env.agents[0]).n), (2, 1)) 
            ac_in_0 = (
                obs_batch[jnp.newaxis, :],
                last_done,
                avail_actions[jnp.newaxis, :],
                out_permutations
            )
            network_0 = fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_network
        else:
            raise ValueError(f"Invalid network type: {network_type_0}")

        if network_type_1 == 'vanilla':
            ac_in_1 = (
                obs_batch[jnp.newaxis, :],
                last_done,
                avail_actions[jnp.newaxis, :],
            )
            network_1 = ippo_ff_hanabi_network
        elif network_type_1 == 'alt_indices':
            ac_in_1 = (
                obs_batch[jnp.newaxis, :],
                last_done,
                avail_actions[jnp.newaxis, :],
            )
            network_1 = fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_alt_network
        elif network_type_1 == 'gfppoy':
            out_permutations = jnp.tile(jnp.identity(env.action_space(env.agents[0]).n), (2, 1, 1)) 
            ac_in_1 = (
                obs_batch[jnp.newaxis, :],
                last_done,
                avail_actions[jnp.newaxis, :],
                out_permutations
            )
            network_1 = decpomdp_symmetries_op_ippo_ff_hanabi_network
        elif network_type_1 == 'gfppoy_indices':
            out_permutations = jnp.tile(jnp.arange(env.action_space(env.agents[0]).n), (2, 1)) 
            ac_in_1 = (
                obs_batch[jnp.newaxis, :],
                last_done,
                avail_actions[jnp.newaxis, :],
                out_permutations
            )
            network_1 = fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_network
        else:
            raise ValueError(f"Invalid network type: {network_type_1}")

        # Compute actions for both models
        pi0, _ = network_0.apply(network_params_0, ac_in_0)
        pi1, _ = network_1.apply(network_params_1, ac_in_1)

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
        return (cum_rew, t, done, rng, env_state, obs, legal_moves)

    init_val = (0, 0, False, rng, env_state, obs, legal_moves)
    cum_rew, _, _, _, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)

    return cum_rew



play_games = jax.vmap(play_game, in_axes=(0, None, None, None, None))


# %%

# def play_a_pair_of_samples(sample_0: dict, sample_1: dict):
#     sample_0_seeds = sorted(list(sample_0.keys()))
#     sample_1_seeds = sorted(list(sample_1.keys()))
#     print(f"Evaluating seeds {sample_0_seeds} vs seeds {sample_1_seeds}")

#     rng = jax.random.PRNGKey(0)
#     xp = jnp.zeros((len(sample_0_seeds), len(sample_1_seeds)))
#     for i, seed_0 in enumerate(sample_0_seeds):
#         for j, seed_1 in enumerate(sample_1_seeds):
#             print(f"Evaluating seed {seed_0} vs seed {seed_1}")
#             rng, _rng = jax.random.split(rng, 2)
#             game_rngs = jax.random.split(_rng, 100)
#             xp = xp.at[i, j].set(
#                 jnp.mean(
#                     play_games(
#                         game_rngs,
#                         sample_0[seed_0],
#                         sample_1[seed_1],
#                         network_type_0 = 'vanilla',
#                         network_type_1 = 'vanilla',
#                     )
#                 )
#             )
#             print(f"XP[{i}, {j}] = {xp[i, j]}")
#     return xp

# %%
vmapped_play_games = jax.vmap(jax.vmap(play_games, in_axes=(None, 0, None, None, None)), in_axes=(None, None, 0, None, None))
# %%
# def play_all_samples(samples: list[dict]):
#     xxp = jnp.zeros((len(samples), len(samples)))
#     for i, sample_i in enumerate(samples):
#         for j, sample_j in enumerate(samples):
#             xp = play_a_pair_of_samples(sample_i, sample_j)
#             avg_xp = jnp.mean(xp)
#             xxp = xxp.at[i, j].set(avg_xp)
    # return xxp

xp_results = {}

# %%
def vmapped_play_all_samples(sample_names: list[str], samples_batched: list[dict], samples_types: list[str]):
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng, 2)
    game_rngs = jax.random.split(_rng, 5000)
    xxp = jnp.zeros((len(samples_batched), len(samples_batched)))
    for i, sample_i in enumerate(samples_batched):
        for j, sample_j in enumerate(samples_batched):
            if False:
            # if (sample_names[i], sample_names[j]) in xp_results:
                xp = xp_results[(sample_names[i], sample_names[j])]
                print(f"Using cached results for {sample_names[i]} vs {sample_names[j]}")
            else:
                print(f"Evaluating {sample_names[i]} vs {sample_names[j]}")
                # import pdb; pdb.set_trace()
                xp = jax.block_until_ready(vmapped_play_games(game_rngs, sample_i, sample_j, samples_types[i], samples_types[j]))
                xp = jnp.mean(xp, axis=2)
                xp_results[(sample_names[i], sample_names[j])] = np.array(xp)
            if i == j:
                xp_mask = ~jnp.eye(xp.shape[0], dtype=bool)
                avg_xp = jnp.mean(xp[xp_mask])
            else:
                avg_xp = jnp.mean(xp)
            print(avg_xp)
            xxp = xxp.at[i, j].set(avg_xp)
    return xxp

def vmapped_play_all_samples_full_data(sample_names: list[str], samples_batched: list[dict], samples_types: list[str]):
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng, 2)
    game_rngs = jax.random.split(_rng, 5000)
    xxp = jnp.zeros((len(samples_batched), len(samples_batched), 5, 5, len(game_rngs)))
    for i, sample_i in enumerate(samples_batched):
        for j, sample_j in enumerate(samples_batched):
            if False:
            # if (sample_names[i], sample_names[j]) in xp_results:
                xp = xp_results[(sample_names[i], sample_names[j])]
                print(f"Using cached results for {sample_names[i]} vs {sample_names[j]}")
            else:
                print(f"Evaluating {sample_names[i]} vs {sample_names[j]}")
                # import pdb; pdb.set_trace()
                xp = jax.block_until_ready(vmapped_play_games(game_rngs, sample_i, sample_j, samples_types[i], samples_types[j]))
                # xp = jnp.mean(xp, axis=2)
                # xp_results[(sample_names[i], sample_names[j])] = np.array(xp)
            # if i == j:
            #     xp_mask = ~jnp.eye(xp.shape[0], dtype=bool)
            #     avg_xp = jnp.mean(xp[xp_mask])
            # else:
            #     avg_xp = jnp.mean(xp)
            # print(avg_xp)
            xxp = xxp.at[i, j, :, :, :].set(xp)
    return xxp




# %%
# show the loaded params and the seeds of each sample
samples_and_seeds = {}
samples_and_n_seeds = {}
for sample_name, sample_seeds in loaded_params.items():
    samples_and_seeds[sample_name] = list(sample_seeds.keys())
    samples_and_n_seeds[sample_name] = len(list(sample_seeds.keys()))
print(samples_and_seeds)
print(samples_and_n_seeds)

# %%

# samples_for_xxp = ['20250910_162923_847628_pmapped', 'decpomdp_symmetries_op_ippo_ff_hanabi_pmapped', '20250915_232735_502690', '20250916_203332_786470', '20250916_213559_958721']
# samples_for_xxp_types = ['vanilla', 'gfppoy', 'vanilla', 'vanilla', 'vanilla']
# samples_for_xxp_batched = [jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *list(loaded_params[sample].values())) for sample in samples_for_xxp]

# # %%
# samples = samples_for_xxp_batched
# start_time = time.time()
# xxp = jax.block_until_ready(vmapped_play_all_samples(samples_for_xxp, samples_for_xxp_batched, samples_for_xxp_types))
# end_time = time.time()
# print(f"Time taken: {end_time - start_time} seconds")
# print(samples_for_xxp)
# print(xxp)

# %%

loaded_params_best_of_k = {}
for sample_name, sample_seeds in loaded_params.items():
    loaded_params_best_of_k[sample_name] = {}
    for seed, params in sample_seeds.items():
        if seed in sample_best_of_k_seeds[sample_name]:
            loaded_params_best_of_k[sample_name][seed] = params
        else:
            pass

best_of_k_samples_and_seeds = {}
best_of_k_samples_and_n_seeds = {}
for sample_name, sample_seeds in loaded_params_best_of_k.items():
    best_of_k_samples_and_seeds[sample_name] = list(sample_seeds.keys())
    best_of_k_samples_and_n_seeds[sample_name] = len(list(sample_seeds.keys()))
print(best_of_k_samples_and_seeds)
print(best_of_k_samples_and_n_seeds)
# %%
# BEST OF K SEEDS
samples_for_xxp = ['decpomdp_symmetries_op_ippo_ff_hanabi_pmapped', 
                #    'fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped', 
                #    'fast_decpomdp_symmetries_op_ippo_ff_hanabi_pmapped_alt', 
                #    'ippo_ff_hanabi_pmapped'
]
samples_for_xxp_types = [sample_network_types[sample] for sample in samples_for_xxp]
samples_for_best_of_k_xxp_batched = [jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *list(loaded_params_best_of_k[sample].values())) for sample in samples_for_xxp]

# %%
# start_time = time.time()
# xxp_best_of_k = jax.block_until_ready(vmapped_play_all_samples_full_data(samples_for_xxp, samples_for_best_of_k_xxp_batched, samples_for_xxp_types))
# end_time = time.time()
# print(f"Time taken: {end_time - start_time} seconds")
# print(samples_for_xxp)

start_time = time.time()
xxp = jax.block_until_ready(vmapped_play_all_samples(samples_for_xxp, samples_for_best_of_k_xxp_batched, samples_for_xxp_types))
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(samples_for_xxp)
print(xxp)
# %%
# xxp_best_of_k.shape
# # %%
# # %%
# xxp_best_of_k_avged_rollouts = xxp_best_of_k.mean(axis=4)
# xxp_best_of_k_avged_rollouts.shape
# # %%
# xxp_best_of_k_avged_rollouts_avg_xp = xxp_best_of_k_avged_rollouts.mean(axis=(2, 3))
# print(xxp_best_of_k_avged_rollouts_avg_xp)
# # %%
# xxp_best_of_k_avged_rollouts_std_xp = xxp_best_of_k_avged_rollouts.std(axis=(2, 3))
# xxp_best_of_k_avged_rollouts_std_xp
# # %%

# xxp_best_of_k_avged_rollouts.shape
# sxp_mask = jnp.eye(xxp_best_of_k_avged_rollouts.shape[0], dtype=bool)
# xxp_mask = ~sxp_mask
# xxp_mask
# # %%
# xxp_best_of_k_avged_rollouts[sxp_mask].shape
# # %%
# xxp_best_of_k_avged_rollouts[xxp_mask].shape
# # %%
# xxp_best_of_k_avged_rollouts[xxp_mask].mean()
# # %%
# xxp_best_of_k_avged_rollouts[xxp_mask].std()


# # %%
# tol = 1.96 * xxp_best_of_k_avged_rollouts[xxp_mask].std() / jnp.sqrt(xxp_best_of_k_avged_rollouts[xxp_mask].flatten().shape[0])
# tol

# # %%
# xxp_best_of_k_avged_rollouts[sxp_mask][:, xxp_mask].shape

# # %%
# xxp_best_of_k_avged_rollouts[sxp_mask][:, xxp_mask].mean()
# # %%
# xxp_best_of_k_avged_rollouts[sxp_mask][:, xxp_mask].std()
# # %%
# tol = 1.96 * xxp_best_of_k_avged_rollouts[sxp_mask][:, xxp_mask].std() / jnp.sqrt(xxp_best_of_k_avged_rollouts[sxp_mask][:, xxp_mask].flatten().shape[0])
# tol
# # %%

# # %%
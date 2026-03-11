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
# %%
# show all columns when printing a dataframe
pd.set_option('display.max_columns', None)
# %%
repo_root = Path(__file__).parent.parent
sample_dir = repo_root / "samples"
viable_sample_start_time = "2025-07-30 18:00:00"

# Load experiment data
print("Loading experiment data...")
df = read_experiment_to_dataframe("easy_hanabi")
# %%
df.head()

# %%
# get paths of models
model_paths = {}
for sample_name, sample_path in zip(df.sample_name, df.sample_path):
    sample_path = sample_dir / sample_path
    model_paths[sample_name] = list(sample_path.glob("trained_models/model_*.pkl"))
    if len(model_paths[sample_name]) == 0:
        print(f"No models found for {sample_path}")
        model_paths[sample_name] = None
    else:
        print(f"Found {len(model_paths[sample_name])} models for {sample_path}")



# %%
def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}
# dummy network
class ActorCritic(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


# %%
env = jaxmarl.make("hanabi", num_agents=2, num_colors=3, num_ranks=3, hand_size=5, max_info_tokens=3, max_life_tokens=2, num_cards_of_rank=np.array([2, 2, 3]))
network = ActorCritic(action_dim=env.action_space(env.agents[0]).n)
action_dim = env.action_space(env.agents[0]).n
obs_shape = env.observation_space(env.agents[0]).shape
init_x = (
        jnp.zeros((1, 1) + obs_shape),
        jnp.zeros((1, 1), dtype=bool),
        jnp.zeros((1, 1, action_dim)),
    )
rng = jax.random.PRNGKey(0)
init_params = network.init(rng, init_x)
# %%
def evaluate_crossplay(checkpoint_path_0, checkpoint_path_1, n_seeds=1):
    with jax.default_device(jax.devices("cpu")[0]):
        # Load model parameters using flax.serialization
        with open(checkpoint_path_0, 'rb') as f:
            params_bytes_0 = f.read()
        network_params_0 = fs.from_bytes(init_params, params_bytes_0)
        
        with open(checkpoint_path_1, 'rb') as f:
            params_bytes_1 = f.read()
        network_params_1 = fs.from_bytes(init_params, params_bytes_1)

    
    def _step_env(rng, env_state, actions):
        rng, _rng = jax.random.split(rng)
        new_obs, new_env_state, reward, dones, infos = env.step(
            _rng, env_state, actions
        )
        new_legal_moves = env.get_legal_moves(new_env_state)
        return rng, new_env_state, new_obs, reward, dones, new_legal_moves

    
    def _sample_action(network_params, obs, done, legal_moves, rng):
        obs_batch = batchify(obs, env.agents, 2)# obs_batch is a (2, obs_dim) array
        legal_moves_batch = batchify(legal_moves, env.agents, 2)# legal_moves_batch is a (2, num_actions) array
        pi, value = network.apply(network_params, (obs_batch, done, legal_moves_batch)) # pi is a (2, num_actions) array
        action = pi.sample(seed=rng) # pi is a distrubtion for each actor so when we sample from it we get a (num_actors) array of actions. We select the first actor's action.
        return action

    for seed in range(n_seeds):

        rng = jax.random.PRNGKey(seed)
        rng, _rng = jax.random.split(rng)
        

        obs, env_state = env.reset(_rng) 
        legal_moves = env.get_legal_moves(env_state) # legal_moves is a dict of with keys "agent_0" and "agent_1" and values are the legal moves for each agent

        

        done = False
        cum_rew = 0
        t = 0

        # print("\n" + "=" * 40 + "\n")

        while not done:
            # env.render(env_state)

            rng, _rng = jax.random.split(rng)
            action_0 = _sample_action(network_params_0, obs, done, legal_moves, _rng)[0]

            rng, _rng = jax.random.split(rng)
            action_1 = _sample_action(network_params_1, obs, done, legal_moves, _rng)[1]

            actions = {
                "agent_0": action_0,
                "agent_1": action_1
            }

        
            rng, env_state, obs, reward, dones, legal_moves = _step_env( # step_env accepts a dict of actions for each agent
                rng, env_state, actions
            )

            done = dones["__all__"]
            cum_rew += reward["__all__"]
            t += 1

            # print("\n" + "=" * 40 + "\n")

    return cum_rew

model_paths = {k: v for k, v in model_paths.items() if v is not None}
model_paths = {k: v for k, v in model_paths.items() if len(v) > 0}
# %%
crossplay_results = np.zeros((len(model_paths), len(model_paths)))
for i, (sample_name_0, model_paths_0) in enumerate(model_paths.items()):
    for j, (sample_name_1, model_paths_1) in enumerate(model_paths.items()):
        rewards = []
        print(f"{sample_name_0} vs {sample_name_1}")
        if model_paths_0 is None or model_paths_1 is None:
            print("No models found")
            crossplay_results[i, j] = np.nan
            continue
        for model_path_0, model_path_1 in zip(model_paths_0, model_paths_1):
            try:
                cum_rew = evaluate_crossplay(model_path_0, model_path_1)
                rewards.append(cum_rew)
            except Exception as e:
                print(f"Error evaluating crossplay for {sample_name_0} vs {sample_name_1}: {e}")
                crossplay_results[i, j] = np.nan

        crossplay_results[i, j] = np.mean(rewards)

# %%
crossplay_results
# %%


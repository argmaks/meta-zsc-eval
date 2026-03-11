# %%
import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl
import os
from omegaconf import OmegaConf
import flax.serialization as fs
from networks import ActorCritic, batchify, unbatchify
from jaxmarl.wrappers.baselines import LogWrapper
import pickle
from typing import Tuple

def load_permutations() -> Tuple[np.ndarray, np.ndarray]:
    """Load observation and action permutation matrices.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays with shapes
        ``(num_permutations, obs_dim, obs_dim)``  and
        ``(num_permutations, action_dim, action_dim)`` corresponding to
        observation and action permutation matrices, respectively.
    """

    # Get all files in the directory
    directory = os.path.join(os.path.dirname(__file__), 'hanabi_symmetries')
    with open(os.path.join(directory, 'obs_permutation_matrices.pkl'), 'rb') as f:
        obs_permutations_matrices = pickle.load(f)
    with open(os.path.join(directory, 'action_permutation_matrices.pkl'), 'rb') as f:
        action_permutations_matrices = pickle.load(f)

    return obs_permutations_matrices, action_permutations_matrices # obs permutations are applied via v @ P, action permutations are applied via P @ v because they are already an inverse permutation

def evaluate_under_permutations(config, rng, checkpoint_timestep=None, rollouts=1000):

    obs_permutations, action_permutations = load_permutations()

    config["ENV_KWARGS"]["num_cards_of_rank"] = np.array(config["ENV_KWARGS"]["num_cards_of_rank"])

    if checkpoint_timestep is None:
        checkpoint_timestep = config["TOTAL_TIMESTEPS"]

    n_models = config["NUM_SEEDS"]

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env = LogWrapper(env)
    config["NUM_ACTORS"] = env.num_agents * 1

    network = ActorCritic(action_dim=env.action_space(env.agents[0]).n, config=config)
    
    # Initialize network to get parameter structure
    init_x = (
        jnp.zeros((1, rollouts, env.observation_space(env.agents[0]).shape)),
        jnp.zeros((1, rollouts)),
        jnp.zeros((1, rollouts, env.action_space(env.agents[0]).n)),
    )
    rng, _rng = jax.random.split(rng)
    network_params_template = network.init(_rng, init_x)

    # Load model parameters using flax.serialization
    network_params_list = []
    for i in range(n_models):
        model_path = os.path.join(os.path.dirname(__file__), "models", str(checkpoint_timestep), f"model_{i}.pkl")
        with open(model_path, 'rb') as f:
            params_bytes = f.read()
        params = fs.from_bytes(network_params_template, params_bytes)
        network_params_list.append(params)

    assert len(network_params_list) == n_models, "Failed to load one or more checkpoints."

    def transform_obs(obs, in_permutation):
        transformed_obs = jnp.dot(obs, in_permutation)
        return transformed_obs

    @jax.jit
    def play_game(rng, network_params_0, network_params_1):

        # Single-environment rollout using the loaded network parameters
        rng, _rng = jax.random.split(rng)

        obs, env_state = env.reset(_rng)
        legal_moves = env.get_legal_moves(env_state)

        local_num_actors = env.num_agents  # 2 for Hanabi two-player

        rng, _rng = jax.random.split(rng)
        shuffle_colour_indices = jax.random.choice(_rng, obs_permutations.shape[0], shape=(config["NUM_ACTORS"],), replace=True)
        # shuffle_colour_indices = jnp.array([0] * config["NUM_ACTORS"]) # FOR DEBUG

        obs_perms = jnp.array(obs_permutations).at[shuffle_colour_indices].get()
        action_perms = jnp.array(action_permutations).at[shuffle_colour_indices].get()

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


            # PERMUTE AVAILABILITY MASKS
            avail_actions = jnp.einsum('bi,bij->bj', avail_actions, action_perms.transpose(0, 2, 1))

            obs_batch = batchify(obs, env.agents, local_num_actors)

            # jax.debug.breakpoint()

            # PERMUTE OBSERVATIONS
            obs_batch = jax.vmap(transform_obs, in_axes=(0, 0))(
                    obs_batch.reshape(-1, obs_batch.shape[-1]),
                    obs_perms
                ).reshape(obs_batch.shape)

            # jax.debug.breakpoint()


            last_done = jnp.zeros((1, local_num_actors), dtype=bool)

            ac_in = (
                obs_batch[jnp.newaxis, :],
                last_done,
                avail_actions[jnp.newaxis, :],
            )

            # Compute actions for both models
            pi0, _ = network.apply(network_params_0, ac_in)
            pi1, _ = network.apply(network_params_1, ac_in)

            rng0, rng1 = jax.random.split(_rng)
            action0 = pi0.sample(seed=rng0)
            action1 = pi1.sample(seed=rng1)

            # UNPERMUTE ACTIONS
            action0 = action0.squeeze()
            action1 = action1.squeeze()
            num_actions = env.action_space(env.agents[0]).n
            one_hot0 = jax.nn.one_hot(action0, num_actions)
            one_hot1 = jax.nn.one_hot(action1, num_actions)
            orig_one_hot0 = jnp.einsum('bi,bij->bj', one_hot0, action_perms) # note action_perms is already an inverse permutation
            orig_one_hot1 = jnp.einsum('bi,bij->bj', one_hot1, action_perms)

            action0 = jnp.argmax(orig_one_hot0, axis=-1)
            action1 = jnp.argmax(orig_one_hot1, axis=-1)

            
    

            env_act0 = unbatchify(action0, env.agents, 1, env.num_agents)
            env_act0 = jax.tree.map(lambda x: x.squeeze(), env_act0)
            env_act1 = unbatchify(action1, env.agents, 1, env.num_agents)
            env_act1 = jax.tree.map(lambda x: x.squeeze(), env_act1)

            # import pdb; pdb.set_trace()

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


    
    play_games = jax.jit(jax.vmap(play_game, in_axes=(0, None, None)))

    # def play_games(game_rngs, network_params_0, network_params_1):
    #     for i, game_rng in enumerate(game_rngs):
    #         returns = jnp.zeros(len(game_rngs))
    #         r = play_game(game_rng, network_params_0, network_params_1)
    #         returns = returns.at[i].set(r)
    #     return returns



    # import pdb; pdb.set_trace() 
    # unroll_env(network_params_list[0], network_params_list[1], rng)

    # vmap over all pairs of models
    network_params_batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *network_params_list) # convert the list of pytrees into a pytree of arrays
    # import pdb; pdb.set_trace()
    # with jax.disable_jit():
    # xp = jax.jit(jax.vmap(jax.vmap(unroll_env, in_axes=(0, None, None)), in_axes=(None, 0, None)))(network_params_batched, network_params_batched, rng)
    returns =jnp.zeros((n_models))
    for i in range(n_models):
            rng, _rng = jax.random.split(rng, 2)
            game_rngs = jax.random.split(_rng, rollouts)
            returns = returns.at[i].set(jnp.mean(play_games(game_rngs, network_params_list[i], network_params_list[i])))

    return returns

# %%
if __name__ == "__main__":
    # %%
    import time
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config)
    config = config["runner"]["algo_config"]
    rng = jax.random.PRNGKey(1)
    start_time = time.time()
    returns = evaluate_under_permutations(config, rng)
    print(np.array(returns).round(1))
    print("avg self-play score under permutations: ", returns.mean())
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
 
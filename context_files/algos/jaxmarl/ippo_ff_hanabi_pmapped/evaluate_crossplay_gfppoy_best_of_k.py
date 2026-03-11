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

def evaluate_crossplay(config, rng, checkpoint_timestep=None, rollouts=5000):

    config["ENV_KWARGS"]["num_cards_of_rank"] = np.array(config["ENV_KWARGS"]["num_cards_of_rank"])

    if checkpoint_timestep is None:
        checkpoint_timestep = config["TOTAL_TIMESTEPS"]

    seeds = config["SEEDS"]
    n_models = len(seeds)

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env = LogWrapper(env)
    config["NUM_ACTORS"] = env.num_agents * 1

    network = ActorCritic(action_dim=env.action_space(env.agents[0]).n, config=config)
    
    # Initialize network to get parameter structure
    init_x = (
        jnp.zeros((1, 1, env.observation_space(env.agents[0]).shape)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1, env.action_space(env.agents[0]).n))
    )
    rng, _rng = jax.random.split(rng)
    network_params_template = network.init(_rng, init_x)

    # Load model parameters using flax.serialization
    network_params_list = []
    for seed in seeds:
        model_path = os.path.join(os.path.dirname(__file__), "models", str(checkpoint_timestep), f"model_{seed}.pkl")
        with open(model_path, 'rb') as f:
            params_bytes = f.read()
        params = fs.from_bytes(network_params_template, params_bytes)
        network_params_list.append(params)

    assert len(network_params_list) == n_models, "Failed to load one or more checkpoints."

    @jax.jit
    def play_game(rng, network_params_0, network_params_1):

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


    
    play_games = jax.vmap(play_game, in_axes=(0, None, None))


    # import pdb; pdb.set_trace() 
    # unroll_env(network_params_list[0], network_params_list[1], rng)

    # vmap over all pairs of models
    network_params_batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *network_params_list) # convert the list of pytrees into a pytree of arrays
    # import pdb; pdb.set_trace()
    # with jax.disable_jit():
    # xp = jax.jit(jax.vmap(jax.vmap(unroll_env, in_axes=(0, None, None)), in_axes=(None, 0, None)))(network_params_batched, network_params_batched, rng)
    xp=jnp.zeros((len(seeds), len(seeds)))
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            rng, _rng = jax.random.split(rng, 2)
            game_rngs = jax.random.split(_rng, rollouts)
            xp = xp.at[i, j].set(jnp.mean(play_games(game_rngs, network_params_list[seeds[i]], network_params_list[seeds[j]])))

    return xp

def best_of_k_evaluation(xp, k):

    sp = xp.diagonal()
    n_seeds = xp.shape[0]
    best_of_k_indices = jnp.zeros((n_seeds // k), dtype=int)
    for i in range(n_seeds // k):
        best_of_k_indices = best_of_k_indices.at[i].set(i*k + jnp.argmax(sp[i*k:(i+1)*k]))
    print("Seed indices selected for best of k evaluation: ", best_of_k_indices)
    best_of_k_xp = xp[jnp.ix_(best_of_k_indices, best_of_k_indices)]
    return best_of_k_xp


# %%
if __name__ == "__main__":
    # %%
    import time
    np.set_printoptions(suppress=True, precision=2)
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config)
    config = config["runner"]["algo_config"]
    rng = jax.random.PRNGKey(1)
    start_time = time.time()
    xp = evaluate_crossplay(config, rng)
    print(jnp.round(xp, 2))
    print("avg self-play score: ", xp.diagonal().mean())
    # Mask out diagonal elements and compute mean of cross-play scores
    mask = ~jnp.eye(xp.shape[0], dtype=bool)
    cross_play_scores = xp[mask]
    print("avg cross-play score: ", cross_play_scores.mean())
    best_of_k_xp = best_of_k_evaluation(xp, 3)

    print("best of k crossplay matrix: \n", best_of_k_xp)
    print("avg best of k self-play score: ", best_of_k_xp.diagonal().mean())
    mask = ~jnp.eye(best_of_k_xp.shape[0], dtype=bool)
    best_of_k_cross_play_scores = best_of_k_xp[mask]
    print("avg best of k cross-play score: ", best_of_k_cross_play_scores.mean())
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
 
    
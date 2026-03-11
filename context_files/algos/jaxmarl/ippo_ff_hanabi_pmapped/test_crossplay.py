import flax.linen as nn
import flax.serialization as fs
import jax.numpy as jnp
from typing import Sequence, Dict
import numpy as np
from flax.linen.initializers import constant, orthogonal
import distrax
import hydra
import jaxmarl
import jax
import os
from functools import partial
from omegaconf import OmegaConf
import warnings
from ippo_ff_hanabi import batchify, unbatchify, ActorCritic
from jaxmarl.wrappers.baselines import LogWrapper

# assert that the trained models exist
assert os.path.exists(os.path.join(os.path.dirname(__file__), "models", "100000.0", "model_0.pkl")), "Saved parameters for model 0 do not exist"
assert os.path.exists(os.path.join(os.path.dirname(__file__), "models", "100000.0", "model_1.pkl")), "Saved parameters for model 1 do not exist"

def evaluate_crossplay(config, rng, rollouts=1000):

    config = OmegaConf.to_container(config)
    config["ENV_KWARGS"]["num_cards_of_rank"] = np.array(config["ENV_KWARGS"]["num_cards_of_rank"])
    
    n_models = 2

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)
    config["NUM_ACTORS"] = env.num_agents * rollouts

    network = ActorCritic(action_dim=env.action_space(env.agents[0]).n, config=config)
    
    # Initialize network to get parameter structure
    init_x = (
        jnp.zeros((1, 1, env.observation_space(env.agents[0]).shape)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1, env.action_space(env.agents[0]).n))
    )
    network_params_template = network.init(rng, init_x)

    # Load model parameters using flax.serialization
    network_params_list = []
    for i in range(n_models):
        model_path = os.path.join(os.path.dirname(__file__), "models", "100000.0", f"model_{i}.pkl")
        with open(model_path, 'rb') as f:
            params_bytes = f.read()
        params = fs.from_bytes(network_params_template, params_bytes)
        network_params_list.append(params)

    assert len(network_params_list) == n_models, "Failed to load one or more checkpoints."



    @jax.jit
    def unroll_env(network_params_0, network_params_1, rng):

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, rollouts)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
        done_batch = jnp.zeros((config["NUM_ACTORS"]), dtype=bool)

        rng, _rng = jax.random.split(rng)
        runner_state = (env_state, obsv, done_batch, _rng)

        def _env_step(runner_state, unused):
            env_state, last_obs, last_done, rng = runner_state

            # (Pdb) last_obs
            # {'agent_0': Traced<ShapedArray(float32[1024,657])>with<DynamicJaxprTrace>, 'agent_1': Traced<ShapedArray(float32[1024,657])>with<DynamicJaxprTrace>}

            # (Pdb) last_done
            # Traced<ShapedArray(bool[1024])>with<DynamicJaxprTrace>

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(
                batchify(avail_actions, env.agents, config["NUM_ACTORS"])
            )                                                                   # Traced<ShapedArray(float32[2048,21])>with<DynamicJaxprTrace>
            obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"]) # Traced<ShapedArray(float32[2048,657])>with<DynamicJaxprTrace>
            ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions[np.newaxis, :]) # (Traced<ShapedArray(float32[1,2048,657])>with<DynamicJaxprTrace>, Traced<ShapedArray(bool[1,1024])>with<DynamicJaxprTrace>, Traced<ShapedArray(float32[1,2048,21])>with<DynamicJaxprTrace>)

            pi0, value0 = network.apply(network_params_0, ac_in)
            pi1, value1 = network.apply(network_params_1, ac_in)

            rng0, rng1 = jax.random.split(_rng)
            action0 = pi0.sample(seed=rng0) # Traced<ShapedArray(int32[1,2048])>with<DynamicJaxprTrace>
            env_act0 = unbatchify(action0, env.agents, rollouts, env.num_agents) 
            env_act0 = jax.tree.map(lambda x: x.squeeze(), env_act0) # {'agent_0': Traced<ShapedArray(int32[1024])>with<DynamicJaxprTrace>, 'agent_1': Traced<ShapedArray(int32[1024])>with<DynamicJaxprTrace>}
            action1 = pi1.sample(seed=rng1)
            env_act1 = unbatchify(action1, env.agents, rollouts, env.num_agents)
            env_act1 = jax.tree.map(lambda x: x.squeeze(), env_act1)

            env_act = {
                "agent_0": env_act0[env.agents[0]],
                "agent_1": env_act1[env.agents[1]]
            }

            # import pdb; pdb.set_trace()    

            env_act = jax.tree.map(lambda x: x.squeeze(), env_act)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, rollouts)
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                rng_step, env_state, env_act
            )
            info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
            done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()

            runner_state = (env_state, obsv, done_batch, rng)
            return runner_state, info

        runner_state, infos = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )

        return infos["returned_episode_returns"][-1, :].mean()

    # import pdb; pdb.set_trace() 

    # vmap over all pairs of models
    network_params_batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *network_params_list) # convert the list of pytrees into a pytree of arrays

    xp = jax.jit(jax.vmap(jax.vmap(unroll_env, in_axes=(0, None, None)), in_axes=(None, 0, None)))(network_params_batched, network_params_batched, rng)

    return xp



if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = OmegaConf.load(config_path)
    algo_config = config['runner']['algo_config']
    rng = jax.random.PRNGKey(0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        xp = evaluate_crossplay(algo_config, rng, rollouts=10)

    print(xp)
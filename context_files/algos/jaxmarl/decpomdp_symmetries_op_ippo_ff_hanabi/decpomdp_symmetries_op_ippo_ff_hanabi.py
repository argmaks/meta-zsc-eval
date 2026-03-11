# edited from https://github.com/gfppoy/expected-return-symmetries/

"""
Based on PureJaxRL Implementation of PPO
"""

import os
import pickle

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization as fs
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, List, Tuple
from flax.training.train_state import TrainState
import distrax
from jaxmarl.wrappers.baselines import LogWrapper
# from jaxmarl.environments.hanabi.hanabi_obl import HanabiOBL as HanabiGame # HanabiOBL is just a renamed HanabiEnv
import jaxmarl
import wandb
import functools
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import copy
from networks import ActorCritic, batchify, unbatchify
from evaluate_crossplay_gfppoy import evaluate_crossplay
import orbax.checkpoint as ocp

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

    return obs_permutations_matrices, action_permutations_matrices




class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    shuffle_colour_indices: jnp.ndarray



def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env = HanabiGame(num_agents=3)
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] 
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng, in_permutations, out_permutations, init_train_state=None, num_timesteps=None):

        if num_timesteps is None:
            num_updates = config["NUM_UPDATES"]
        else:
            num_updates = num_timesteps // config["NUM_STEPS"] // config["NUM_ENVS"]

        # INIT NETWORK
        network = ActorCritic(env.action_space(env.agents[0]).n, config=config)
        
        rng, _rng, __rng = jax.random.split(rng, 3)

        shuffle_colour_indices = jax.random.choice(__rng, in_permutations.shape[0], shape=(config["NUM_ENVS"],), replace=True)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
            out_permutations[shuffle_colour_indices]
        )
       
        network_params = network.init(_rng, init_x)
  
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        if init_train_state is None:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        else:
            train_state = init_train_state

        # INIT ENV
        rng, _rng, __rng = jax.random.split(rng, 3)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
        shuffle_colour_indices = jax.random.choice(__rng, in_permutations.shape[0], shape=(config["NUM_ACTORS"],), replace=True)

        def transform_obs(obs, in_permutation):
            transformed_obs = jnp.dot(obs, in_permutation)
            return transformed_obs

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, shuffle_colour_indices, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                obs_batch = jax.vmap(transform_obs, in_axes=(0, 0))(
                    obs_batch.reshape(-1, obs_batch.shape[-1]),
                    in_permutations[shuffle_colour_indices]
                ).reshape(obs_batch.shape)

                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions[np.newaxis, :], out_permutations[shuffle_colour_indices])
                pi, value = network.apply(train_state.params, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                env_act = jax.tree.map(lambda x: x.squeeze(), env_act)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                
                def _resample_indices(rng, shuffle_colour_indices, done_batch, max_index):
                    rng, _rng = jax.random.split(rng)

                    new_indices = jax.random.choice(_rng, max_index, shape=shuffle_colour_indices.shape, replace=True)

                    updated_shuffle_colour_indices = jnp.where(done_batch, new_indices, shuffle_colour_indices)

                    return rng, updated_shuffle_colour_indices

                rng, updated_shuffle_colour_indices = _resample_indices(rng, shuffle_colour_indices, done_batch, in_permutations.shape[0])

                transition = Transition(
                    done_batch,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                    shuffle_colour_indices
                )
                runner_state = (train_state, env_state, obsv, done_batch, updated_shuffle_colour_indices, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, shuffle_colour_indices, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_obs_batch = jax.vmap(transform_obs, in_axes=(0, 0))(
                    last_obs_batch.reshape(-1, last_obs_batch.shape[-1]),
                    in_permutations[shuffle_colour_indices],
            ).reshape(last_obs_batch.shape)
            avail_actions = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).n)
            )
            ac_in = (
                last_obs_batch[np.newaxis, :], 
                last_done[np.newaxis, :], 
                avail_actions, 
                out_permutations[shuffle_colour_indices]
            )
            _, last_val = network.apply(train_state.params, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK

                        pi, value = network.apply(params,
                                                  (traj_batch.obs, traj_batch.done, traj_batch.avail_actions, out_permutations[traj_batch.shuffle_colour_indices.ravel()]))
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                        **metric["loss"],
                    }
                )
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, shuffle_colour_indices, rng)
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ACTORS"]), dtype=bool), shuffle_colour_indices, _rng)
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, num_updates
        )
        return {"runner_state": runner_state}

    return train


def create_dummy_train_state(config):
    rng = jax.random.PRNGKey(0)
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac
    network = ActorCritic(env.action_space(env.agents[0]).n, config=config)
    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape)
        ),
        jnp.zeros((1, config["NUM_ENVS"])),
        jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        jnp.tile(jnp.identity(env.action_space(env.agents[0]).n), (config["NUM_ENVS"], 1, 1)) # dummy for out_permutations
    )
    network_params = network.init(_rng, init_x)
    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
    dummy_train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    return dummy_train_state




def main(config):

    raw_config = OmegaConf.to_container(config)
    config = copy.deepcopy(raw_config)
    if config["ENV_KWARGS"]["num_cards_of_rank"] is not None:
        config["ENV_KWARGS"]["num_cards_of_rank"] = np.array(config["ENV_KWARGS"]["num_cards_of_rank"])

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_save_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_save_dir, exist_ok=True)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    in_permutations, out_permutations = load_permutations()
    config["NUM_PERMUTATIONS"] = in_permutations.shape[0]

    # Setup random number generation for reproducibility
    rng = jax.random.PRNGKey(config["SEED"])

    # Setup checkpoint managers for each seed
    ckpt_managers = []
    options = ocp.CheckpointManagerOptions(create=True)
    for i in range(config["NUM_SEEDS"]):
        ckpt_manager = ocp.CheckpointManager(
            os.path.join(checkpoint_dir, f"seed_{i}"),
            options=options
        )
        ckpt_managers.append(ckpt_manager)

    # Run training across multiple seeds in parallel for statistical significance
    rngs = jax.random.split(rng, config["NUM_SEEDS"])


    if config["CHECKPOINT_TIMESTEPS"] is not None:
        checkpoint_timesteps = config["CHECKPOINT_TIMESTEPS"] + [config["TOTAL_TIMESTEPS"]]
        for t, checkpoint_timestep in enumerate(checkpoint_timesteps):

            if t == 0:
                train_vmap_jit = jax.jit(
                    jax.vmap(make_train(config), in_axes=(0, None, None, None, None)),
                    static_argnames=["num_timesteps"],
                )
                target_timesteps = int(checkpoint_timestep)
                outs = jax.block_until_ready(train_vmap_jit(rngs, in_permutations, out_permutations, None, target_timesteps))
            else:
                # restore each seed
                dummy_train_state = create_dummy_train_state(config)
                restored_train_states = []
                for i, rng in enumerate(rngs):
                    restored_train_states.append(ckpt_managers[i].restore(int(checkpoint_timesteps[t-1]), args=ocp.args.StandardRestore(dummy_train_state)))

                # batchify  
                checkpoint_train_states = jax.tree_util.tree_map(
                    lambda *xs: jnp.stack(xs),  # stack along a new first axis
                    *restored_train_states
                )
                train_vmap_jit = jax.jit(
                    jax.vmap(make_train(config), in_axes=(0, None, None, 0, None)),
                    static_argnames=["num_timesteps"],
                )
                target_timesteps = int(checkpoint_timestep - checkpoint_timesteps[t-1])
                outs = jax.block_until_ready(train_vmap_jit(rngs, in_permutations, out_permutations, checkpoint_train_states, target_timesteps))

            # checkpoint each seed
            for i, rng in enumerate(rngs):
                train_state = jax.tree.map(lambda x: x[i], outs["runner_state"][0][0])
                ckpt_managers[i].save(int(checkpoint_timestep), args=ocp.args.StandardSave(train_state))
                ckpt_managers[i].wait_until_finished()

            # save model parameters
            for i, rng in enumerate(rngs):
                model_state = outs["runner_state"][0][0]
                params = jax.tree.map(lambda x: x[i], model_state.params)
                params_bytes = fs.to_bytes(params)
                model_path = os.path.join(model_save_dir, str(checkpoint_timestep), f'model_{i}.pkl')
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, 'wb') as f:
                    f.write(params_bytes)


            # evaluate crossplay
            xp_rng = jax.random.PRNGKey(0)
            xp_matrix = evaluate_crossplay(config, xp_rng, checkpoint_timestep)
            avg_self_play_score = np.mean(np.diag(xp_matrix))
            avg_cross_play_score = np.mean(
                xp_matrix[~np.eye(xp_matrix.shape[0], dtype=bool)]
            )
            self_play_cross_play_diff = avg_self_play_score - avg_cross_play_score
            wandb.log({
                "avg_self_play_score": avg_self_play_score,
                "avg_cross_play_score": avg_cross_play_score,
                "self_play_cross_play_diff": self_play_cross_play_diff
            })
            print(f"At timestep {checkpoint_timestep}, Avg self-play score: {avg_self_play_score}, Avg cross-play score: {avg_cross_play_score}, Self-play cross-play difference: {self_play_cross_play_diff}")
            print(f"Crossplay matrix: {xp_matrix}")
    else:
        train_vmap_jit = jax.jit(
            jax.vmap(make_train(config), in_axes=(0, None, None, None, None)),
            static_argnames=["num_timesteps"],
        )
        target_timesteps = int(config["TOTAL_TIMESTEPS"])
        outs = jax.block_until_ready(train_vmap_jit(rngs, in_permutations, out_permutations, None, target_timesteps))

        # checkpoint each seed
        for i, rng in enumerate(rngs):
            train_state = jax.tree.map(lambda x: x[i], outs["runner_state"][0][0])
            ckpt_managers[i].save(int(config["TOTAL_TIMESTEPS"]), args=ocp.args.StandardSave(train_state))
            ckpt_managers[i].wait_until_finished()

        # save model parameters
        for i, rng in enumerate(rngs):
            model_state = outs["runner_state"][0][0]
            params = jax.tree.map(lambda x: x[i], model_state.params)
            params_bytes = fs.to_bytes(params)
            model_path = os.path.join(model_save_dir, str(config["TOTAL_TIMESTEPS"]), f'model_{i}.pkl')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                f.write(params_bytes)


if __name__ == "__main__":
    main() 
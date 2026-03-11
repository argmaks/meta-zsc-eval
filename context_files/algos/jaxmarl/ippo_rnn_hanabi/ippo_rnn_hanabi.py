"""
Based on PureJaxRL Implementation of PPO

doing homogenous first with continuous actions. Also terminate synchronously

NOTE: currently implemented using the gymnax to smax wrapper
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
import flax.serialization as fs
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
import wandb
import functools
import hydra
from omegaconf import OmegaConf
import os 
import copy
import math
import orbax.checkpoint as ocp
from evaluate_crossplay_gfppoy_best_of_k import evaluate_crossplay, best_of_k_evaluation
from networks import ActorCriticRNN, ScannedRNN, batchify, unbatchify


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray



def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env = GymnaxToSMAX(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env = HanabiGame()
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # env = FlattenObservationWrapper(env) # NOTE need a batchify wrapper
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng, init_train_state=None, num_timesteps=None):

        if num_timesteps is None:
            num_updates = config["NUM_UPDATES"]
        else:
            num_updates = num_timesteps // config["NUM_STEPS"] // config["NUM_ENVS"]

        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n))
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)
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
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
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

                if config["RESET_HSTATE_ON_DONE"]:
                    # reinstatiate hstate for done episodes
                    hstate = jnp.where(done_batch, init_hstate, hstate)
                    
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            init_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            avail_actions = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).n)
            )
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
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
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(params, init_hstate.squeeze(),
                                                     (traj_batch.obs, traj_batch.done, traj_batch.avail_actions))
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
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jnp.reshape(
                    init_hstate, (1, config["NUM_ACTORS"], -1)
                )                
                batch = (init_hstate, traj_batch, advantages.squeeze(), targets.squeeze())
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
                update_state = (train_state, init_hstate.squeeze(), traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
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
                    },
                )
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ACTORS"]), dtype=bool), init_hstate, _rng)
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
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape)
        ),
        jnp.zeros((1, config["NUM_ENVS"])),
        jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n))
    )
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    network_params = network.init(_rng, init_hstate, init_x)
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

# @hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_hanabi")
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
        tags=["IPPO", "RNN", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    # Setup random number generation for reproducibility with pmapping
    seeds = config["SEEDS"]
    n_seeds = len(seeds)

    n_devices = jax.device_count()
    seeds_per_device = math.ceil(n_seeds / n_devices)
    n_total_seeds = seeds_per_device * n_devices
    total_seeds = seeds + [42 + i for i in range(n_total_seeds - n_seeds)]
    rngs = jnp.array([jax.random.PRNGKey(seed) for seed in total_seeds])
    rngs  = rngs.reshape(n_devices, seeds_per_device, 2)

    # Setup checkpoint managers for each seed
    ckpt_managers = []
    options = ocp.CheckpointManagerOptions(create=True)
    for seed in total_seeds:
        ckpt_manager = ocp.CheckpointManager(
            os.path.join(checkpoint_dir, f"seed_{seed}"),
            options=options
        )
        ckpt_managers.append(ckpt_manager)



    if config["CHECKPOINT_TIMESTEPS"] is not None:
        checkpoint_timesteps = config["CHECKPOINT_TIMESTEPS"] + [config["TOTAL_TIMESTEPS"]]
        for t, checkpoint_timestep in enumerate(checkpoint_timesteps):
            print(f"Starting training until timestep {checkpoint_timestep}")
            

            if t == 0:
                train_vmap = jax.vmap(make_train(config), in_axes=(0, None, None))
                train_vmap_pmap = jax.pmap(train_vmap, in_axes=(0, None, None), static_broadcasted_argnums=(2,))
                target_timesteps = int(checkpoint_timestep)
                outs = jax.block_until_ready(train_vmap_pmap(rngs, None, target_timesteps))
            else:
                # restore each seed
                dummy_train_state = create_dummy_train_state(config)
                restored_train_states = []
                for i in range(n_total_seeds):
                    restored_train_states.append(ckpt_managers[i].restore(int(checkpoint_timesteps[t-1]), args=ocp.args.StandardRestore(dummy_train_state)))

                # batchify  
                checkpoint_train_states = jax.tree_util.tree_map(
                    lambda *xs: jnp.stack(xs),  # stack along a new first axis
                    *restored_train_states
                )
                checkpoint_train_states = jax.tree.map(lambda x: x.reshape(n_devices, seeds_per_device, *x.shape[1:]), checkpoint_train_states)
                train_vmap = jax.vmap(make_train(config), in_axes=(0, 0, None))
                train_vmap_pmap = jax.pmap(train_vmap, in_axes=(0, 0, None), static_broadcasted_argnums=(2,))
                target_timesteps = int(checkpoint_timestep - checkpoint_timesteps[t-1])
                outs = jax.block_until_ready(train_vmap_pmap(rngs, checkpoint_train_states, target_timesteps))
        
            # checkpoint each seed
            flat_train_states = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), outs["runner_state"][0][0])
            for i in range(n_total_seeds):
                train_state = jax.tree.map(lambda x: x[i], flat_train_states)
                ckpt_managers[i].save(int(checkpoint_timestep), args=ocp.args.StandardSave(train_state))
                ckpt_managers[i].wait_until_finished()

            # save model parameters
            for i in range(n_total_seeds):
                model_state = flat_train_states
                params = jax.tree.map(lambda x: x[i], model_state.params)
                params_bytes = fs.to_bytes(params)
                model_path = os.path.join(model_save_dir, str(checkpoint_timestep), f'model_{total_seeds[i]}.pkl')
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
            k = config["BEST_OF_K"]
            best_of_k_xp = best_of_k_evaluation(xp_matrix, k)
            avg_best_of_k_self_play_score = np.mean(np.diag(best_of_k_xp))
            avg_best_of_k_cross_play_score = np.mean(
                best_of_k_xp[~np.eye(best_of_k_xp.shape[0], dtype=bool)]
            )
            best_of_k_self_play_cross_play_diff = avg_best_of_k_self_play_score - avg_best_of_k_cross_play_score
            wandb.log({
                "avg_self_play_score": avg_self_play_score,
                "avg_cross_play_score": avg_cross_play_score,
                "self_play_cross_play_diff": self_play_cross_play_diff,
                "avg_best_of_k_self_play_score": avg_best_of_k_self_play_score,
                "avg_best_of_k_cross_play_score": avg_best_of_k_cross_play_score,
                "best_of_k_self_play_cross_play_diff": best_of_k_self_play_cross_play_diff
            })
            print(f"At timestep {checkpoint_timestep}, Avg self-play score: {avg_self_play_score}, Avg cross-play score: {avg_cross_play_score}, Self-play cross-play difference: {self_play_cross_play_diff}")
            print(f"Crossplay matrix: {xp_matrix}")
            print(f"Best of k crossplay matrix: {best_of_k_xp}")
            print(f"Avg best of k self-play score: {avg_best_of_k_self_play_score}, Avg best of k cross-play score: {avg_best_of_k_cross_play_score}, Best of k self-play cross-play difference: {best_of_k_self_play_cross_play_diff}")
    else:
        train_vmap = jax.vmap(make_train(config), in_axes=(0, None, None))
        train_vmap_pmap = jax.pmap(train_vmap, in_axes=(0, None, None), static_broadcasted_argnums=(2,))
        target_timesteps = int(config["TOTAL_TIMESTEPS"])
        outs = jax.block_until_ready(train_vmap_pmap(rngs, None, target_timesteps))

        # checkpoint each seed
        flat_train_states = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), outs["runner_state"][0][0])
        for i in range(n_total_seeds):
            train_state = jax.tree.map(lambda x: x[i], flat_train_states)
            ckpt_managers[i].save(int(config["TOTAL_TIMESTEPS"]), args=ocp.args.StandardSave(train_state))
            ckpt_managers[i].wait_until_finished()

        # save model parameters
        for i in range(n_total_seeds):
            model_state = flat_train_states
            params = jax.tree.map(lambda x: x[i], model_state.params)
            params_bytes = fs.to_bytes(params)
            model_path = os.path.join(model_save_dir, str(config["TOTAL_TIMESTEPS"]), f'model_{total_seeds[i]}.pkl')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                f.write(params_bytes)

    





if __name__ == "__main__":
    main() 

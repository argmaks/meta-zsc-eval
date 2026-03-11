import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, Union
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
import wandb
import functools
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import os, sys
import re
import time
import itertools
import shutil
import inspect

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict

# ABC_ means either pytree with all leaves having shape ABC, or a jnp.ndarray whose shape begins with ABC
# T = NUM_STEPS
# K = k
# E = NUM_ENVS
# P = NUM_PLAYERS
# B = BATCH_SIZE = NUM_ACTORS = E * P
# O = OBSERVATION_SPACE_SIZE
# A = ACTION_SPACE_SIZE
# H = HAND_SIZE
# C = NUM_COLORS
# R = NUM_RANKS
# X = C * R
# Y = H * C * R
# Z = (P - 1) * H * C * R
# D = DECK_SIZE
# M = MINIBATCH_SIZE
# W = LAYER_WIDTH
# L = NUM_RNN_LAYERS

class LSTMEncoder(nn.Module):
    hidden_size: int = 512 # TODO replace with config everywhere
    num_ff_layers: int = 2
    num_rnn_layers: int = 2

    @nn.compact
    def __call__(self, hidden_LBW_, x):
        obs_BO, done_B = x
        y_BW = nn.relu(nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs_BO))
        for layer in range(1, self.num_ff_layers):
            y_BW = nn.relu(nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(y_BW))

        hidden_LBW_ = tuple(
            jnp.where(done_B[:, None], self.initialize_hidden_state(self.num_rnn_layers, y_BW.shape[0], self.hidden_size)[i], hidden_LBW_[i])
            for i in range(len(hidden_LBW_))
        )
        cs_LBW, hs_LBW = hidden_LBW_
        new_cs_LBW, new_hs_LBW = jnp.zeros_like(cs_LBW), jnp.zeros_like(hs_LBW)

        for layer in range(self.num_rnn_layers):
            (new_c_BW,  new_h_BW), y_BW = nn.LSTMCell(features=self.hidden_size)(
                (cs_LBW[layer], hs_LBW[layer]), y_BW
            )
            new_cs_LBW = new_cs_LBW.at[layer].set(new_c_BW)
            new_hs_LBW = new_hs_LBW.at[layer].set(new_h_BW)
            
        return (new_cs_LBW, new_hs_LBW), y_BW

    @staticmethod
    def initialize_hidden_state(num_rnn_layers, batch_size, hidden_size):
        single_LBW = jnp.zeros((num_rnn_layers, batch_size, hidden_size))
        return (single_LBW, single_LBW)

class LSTMDecoder(nn.Module):
    card_embed_size: int
    hidden_size: int
    num_colors: int
    num_ranks: int
    hand_size: int
    num_ff_layers: int
    count_based: bool = False

    @nn.compact
    def sample_card(self, rng, hidden_BW_, previous_card_BX, encoded_state_BW, legal_cards_BX):
        probabilities_new_card_count_BX = legal_cards_BX / (legal_cards_BX.sum(axis=-1)[..., None] + 1e-10)
        logits_new_card_count_BX = jnp.log(probabilities_new_card_count_BX + 1e-10)
        if self.count_based:
            logits_new_card_BX = logits_new_card_count_BX
            new_hidden_BW_ = hidden_BW_
        else:
            encoded_previous_card_B_ = nn.Dense(self.card_embed_size, use_bias=False)(previous_card_BX)
            input_to_net_B_ = jnp.concatenate((encoded_previous_card_B_, encoded_state_BW), axis=-1)
            new_hidden_BW_, y_BW = nn.LSTMCell(self.hidden_size)(hidden_BW_, input_to_net_B_)
            for layer in range(self.num_ff_layers):
                y_BW = nn.relu(nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(y_BW))
            logits_new_card_BX = nn.Dense(self.num_colors * self.num_ranks)(y_BW)
        
        logits_new_card_legal_BX = logits_new_card_BX - 1e10 * (legal_cards_BX == 0)

        rng, _rng = jax.random.split(rng)
        sampled_card_B = jax.random.categorical(_rng, logits_new_card_legal_BX, axis=-1)
        sampled_card_BX = jax.nn.one_hot(sampled_card_B, self.num_colors * self.num_ranks)
        sampled_card_BCR = sampled_card_BX.reshape((sampled_card_BX.shape[0], self.num_colors, self.num_ranks))
        return new_hidden_BW_, sampled_card_BCR, logits_new_card_count_BX, logits_new_card_legal_BX

    # This scan is over the card dimension in each hand
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    def __call__(self, carry, step):
        rng, hidden_BW_, remaining_cards_BX = carry # TODO better name for remaining cards
        previous_card_BX, encoded_state_BW, card_knowledge_BX = step

        new_remaining_cards_BX = remaining_cards_BX - previous_card_BX
        legal_cards_BX = new_remaining_cards_BX * card_knowledge_BX

        rng, _rng = jax.random.split(rng)
        new_hidden_BW_, sampled_card_BCR, logits_new_card_count_BX, logits_new_card_legal_BX = self.sample_card(
            _rng, hidden_BW_, previous_card_BX, encoded_state_BW, legal_cards_BX,
        )

        return (
            rng, 
            new_hidden_BW_,
            new_remaining_cards_BX,
        ), (
            logits_new_card_count_BX,
            logits_new_card_legal_BX,
        )

    # This scan is over the card dimension in each hand
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    # Here we sample autoregressively, WITHOUT using teacher forcing
    def ar_sample(self, carry, step):
        rng, hidden_EW_, previous_card_EX, remaining_cards_EX, valid_E = carry
        encoded_state_EW, card_knowledge_EX = step

        legal_cards_EX = remaining_cards_EX * card_knowledge_EX

        new_valid_E = valid_E * (legal_cards_EX.sum(axis=-1) > 0)

        rng, _rng = jax.random.split(rng)
        new_hidden_EW_, sampled_card_ECR, logits_new_card_count_EX, logits_new_card_legal_EX = self.sample_card(
            _rng, hidden_EW_, previous_card_EX, encoded_state_EW, legal_cards_EX,
        )

        sampled_card_EX = sampled_card_ECR.reshape((sampled_card_ECR.shape[0], -1,))
        new_remaining_cards_EX = remaining_cards_EX - sampled_card_EX

        return (
            rng,
            new_hidden_EW_,
            sampled_card_EX,
            new_remaining_cards_EX,
            new_valid_E,
        ), (
            sampled_card_ECR, 
            logits_new_card_count_EX,
            logits_new_card_legal_EX,
        )

    @staticmethod
    def initialize_hidden_state(batch_size, hidden_size):
        single = jnp.zeros((batch_size, hidden_size))
        return (single, single)

class BeliefModel(nn.Module):
    hidden_size: int = 512
    num_colors: int = 5
    num_ranks: int = 5
    hand_size: int = 5
    deck_size: int = 50
    num_rnn_layers: int = 2
    num_ff_layers_encoder: int = 2
    num_ff_layers_decoder: int = 0
    count_based: bool = False

    def setup(self):
        self.encoder = LSTMEncoder(
            self.hidden_size, # replace by layer_width across the whole file
            self.num_ff_layers_encoder,
            self.num_rnn_layers,
        )
        self.decoder = LSTMDecoder(
            self.hidden_size // 8, 
            self.hidden_size, 
            self.num_colors, 
            self.num_ranks,
            self.hand_size,
            self.num_ff_layers_decoder,
            self.count_based,
        )

    def encode(self, hidden_LBW_, obs_BO, done_B):
        if self.count_based:
            encoded_state_BW = jnp.zeros((obs_BO.shape[0], self.hidden_size))
        else:
            hidden_LBW_, encoded_state_BW = self.encoder(hidden_LBW_, (obs_BO, done_B))
        return hidden_LBW_, encoded_state_BW
    
    def decode(self, encoded_state_EW, obs_EO, rng, remaining_cards_EX, card_knowledge_EHX):
        initial_decoder_hidden_EW_ = self.decoder.initialize_hidden_state(obs_EO.shape[0], self.hidden_size)
        initial_card_EX = jnp.zeros((obs_EO.shape[0], self.num_colors * self.num_ranks))
        initial_valid_E = jnp.ones(obs_EO.shape[0])
        encoded_state_EHW = jnp.repeat(encoded_state_EW[:, None], self.hand_size, axis=1)
        
        (
            rng,
            _,
            _,
            cards_in_sampled_deck_EX,
            valid_E,
        ), (
            sampled_hand_EHCR,
            logits_new_card_count_EHX,
            logits_new_card_legal_EHX,
        ) = self.decoder.ar_sample(
            (
                rng, 
                initial_decoder_hidden_EW_, 
                initial_card_EX,
                remaining_cards_EX,
                initial_valid_E,
            ),(
                encoded_state_EHW,
                card_knowledge_EHX,
            ),
        )
        
        def _sample_deck(rng, cards_in_sampled_deck_X):
            cards_in_new_deck_indices_D = jnp.repeat(
                jnp.arange(self.num_colors * self.num_ranks + 1),
                jnp.concatenate([cards_in_sampled_deck_X.astype(jnp.int32), jnp.array([0])]),
                total_repeat_length=self.deck_size
            )

            def _index_to_card(idx):
                card_CR = jnp.zeros((self.num_colors, self.num_ranks))
                return jax.lax.cond(
                    idx < self.num_colors * self.num_ranks,
                    lambda : card_CR.at[jnp.unravel_index(idx, (self.num_colors, self.num_ranks))].set(1),
                    lambda : card_CR
                )

            cards_in_new_deck_DCR = jax.vmap(_index_to_card)(cards_in_new_deck_indices_D)
            cards_in_new_deck_DCR = jax.random.permutation(rng, cards_in_new_deck_DCR)

            nonzero_cards_indices_D = 1 * (0 != cards_in_new_deck_DCR.sum(axis=[1, 2]))
            sort_indices_D = jnp.argsort(nonzero_cards_indices_D, stable=True)
            new_deck_DCR = cards_in_new_deck_DCR[sort_indices_D]

            return new_deck_DCR

        rng_E_ = jax.random.split(rng, obs_EO.shape[0])
        sampled_deck_EDCR = jax.vmap(_sample_deck)(rng_E_, cards_in_sampled_deck_EX)
            
        return valid_E, sampled_hand_EHCR, sampled_deck_EDCR, logits_new_card_count_EHX, logits_new_card_legal_EHX

    # This scan is over the time dimension
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(self, carry, step): # Technically all the B shapes have to replaced by M shapes, as this is called on minibatches
        rng, hidden_LBW_ = carry
        obs_BO, done_B, hand_BY, remaining_cards_BX, card_knowledge_BHX = step

        new_hidden_LBW_, encoded_state_BW = self.encode(hidden_LBW_, obs_BO, done_B)
        encoded_state_BHW = jnp.repeat(encoded_state_BW[:, None], self.hand_size, axis=1)

        initial_decoder_hidden_BW_ = self.decoder.initialize_hidden_state(obs_BO.shape[0], self.hidden_size)

        initial_card_BX = jnp.zeros((obs_BO.shape[0], self.num_colors * self.num_ranks))
        hand_BHX = hand_BY.reshape((hand_BY.shape[0], self.hand_size, -1))
        previous_cards_BHX = jnp.concatenate([initial_card_BX[:, None], hand_BHX[:, :-1]], axis=1)

        # Autoregressively decode the hand, using teacher forcing.
        (rng, _, _), (logits_new_card_count_BHX, logits_new_card_legal_BHX) = self.decoder(
            (
                rng, 
                initial_decoder_hidden_BW_, 
                remaining_cards_BX
            ), (
                previous_cards_BHX,
                encoded_state_BHW,
                card_knowledge_BHX,
            ),
        )
        return (rng, new_hidden_LBW_), (logits_new_card_count_BHX, logits_new_card_legal_BHX)

def sample_hand_from_belief_learned(bm_train_state, rng, env_state_E_, encoded_state_BW, obs_BO, belief_player_idx): # TODO could put all this into BeliefModel.decode??? Or fit decode in here?
    hands_EPHCR = env_state_E_.player_hands
    E, P = hands_EPHCR.shape[:2]
    encoded_state_EW = encoded_state_BW.reshape((P, E, -1)).swapaxes(0, 1)[jnp.arange(E), belief_player_idx]
    obs_EO = obs_BO.reshape((P, E, -1)).swapaxes(0, 1)[jnp.arange(E), belief_player_idx]
    hand_EHCR = hands_EPHCR[jnp.arange(E), belief_player_idx]
    deck_EDCR = env_state_E_.deck
    remaining_cards_EX = deck_EDCR.sum(axis=1).reshape((E, -1)) + hand_EHCR.sum(axis=1).reshape((E, -1))
    card_knowledge_EPHX = env_state_E_.card_knowledge
    card_knowledge_EHX = card_knowledge_EPHX[jnp.arange(E), belief_player_idx]

    valid_E, sampled_hand_EHCR, sampled_deck_EDCR, logits_new_card_count_EHX, logits_new_card_legal_EHX = bm_train_state.apply_fn(
        bm_train_state.params,
        encoded_state_EW, obs_EO, rng, remaining_cards_EX, card_knowledge_EHX,
        method=BeliefModel.decode,
    )
    entropy_count = distrax.Categorical(logits_new_card_count_EHX).entropy().mean()
    entropy_legal = distrax.Categorical(logits_new_card_legal_EHX).entropy().mean()

    valid_E111 = valid_E.reshape(-1, 1, 1, 1)
    hand_to_use_EHCR = hand_EHCR * (1 - valid_E111) + sampled_hand_EHCR * valid_E111
    hands_to_use_EPHCR = hands_EPHCR.at[jnp.arange(E), belief_player_idx].set(hand_to_use_EHCR)
    deck_to_use_EDCR = deck_EDCR * (1 - valid_E111) + sampled_deck_EDCR * valid_E111

    """
    sampled_remaining_cards_EX = sampled_hand_EHCR.sum(axis=1).reshape((E, -1)) + sampled_deck_EDCR.sum(axis=1).reshape((E, -1))
    jax.debug.print("same remaining_cards {}", 0 == jnp.sum(jnp.abs(sampled_remaining_cards_EX != remaining_cards_EX)))
    jax.debug.print("same num_cards_in_hand {}", 0 == jnp.sum(jnp.abs(hand_EHCR.sum(axis=[1,2,3]) != sampled_hand_EHCR.sum(axis=[1,2,3]))))
    jax.debug.print("same num_cards_in_deck {}", 0 == jnp.sum(jnp.abs(deck_EDCR.sum(axis=[1,2,3]) != sampled_deck_EDCR.sum(axis=[1,2,3]))))
    nonzero_cards_ED = 0 != deck_EDCR.sum(axis=[2,3])
    sampled_nonzero_cards_ED = 0 != sampled_deck_EDCR.sum(axis=[2,3])
    jax.debug.print("deck has nonzero cards in same positions {}", 0 == jnp.sum(jnp.abs(nonzero_cards_ED != sampled_nonzero_cards_ED)))
    jax.debug.print("valid_E {}", valid_E.mean())
    hands_to_use_EPHX = hands_to_use_EPHCR.reshape(card_knowledge_EPHX.shape)
    jax.debug.print("no illegal_cards {}", 0 == jnp.sum(hands_to_use_EPHX * (1 - card_knowledge_EPHX)))
    """

    return (
        env_state_E_.replace(player_hands=hands_to_use_EPHCR, deck=deck_to_use_EDCR),
        {
            "valid_percentage_hands": valid_E.mean(),
            "entropy_legal": entropy_legal,
            "entropy_count": entropy_count,
        },
    )

def save_model_checkpoint(train_state, run_name, wandb_run_name, process_id):
    def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
        flattened_dict = flatten_dict(params, sep=",")
        save_file(flattened_dict, filename)

    params = train_state.params

    save_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(save_dir, exist_ok=True)

    def _do_save(params):
        file_path = f"{save_dir}/{run_name}_seed_{process_id}.safetensors"
        save_params(params, file_path)

        script_path = os.path.abspath(inspect.stack()[0].filename)
        shutil.copy(script_path, os.path.join(save_dir, os.path.basename(script_path)))

        # Upload this to wandb as an artifact
        artifact = wandb.Artifact(f"{wandb_run_name}-checkpoint", type="checkpoint")
        artifact.add_file(file_path)
        wandb.log_artifact(artifact, aliases = ["latest"])

    _do_save(params)

class ScannedMultiLayerGRU(nn.Module):
    num_layers: int
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry_LBW, x_B_):
        ins_BW, resets_B = x_B_
        carry_LBW = jnp.where(resets_B[None, :, None], self.initialize_hidden_state(self.num_layers, ins_BW.shape[0], ins_BW.shape[1]), carry_LBW)
        new_carry_LBW = jnp.zeros_like(carry_LBW)
        y_BW = ins_BW

        for layer in range(self.num_layers):
            new_carry_at_layer_BW, y_BW = nn.GRUCell(features=ins_BW.shape[1])(
                carry_LBW[layer], y_BW
            )
            new_carry_LBW = new_carry_LBW.at[layer].set(new_carry_at_layer_BW)

        return new_carry_LBW, y_BW

    @staticmethod
    def initialize_hidden_state(num_rnn_layers, batch_size, hidden_size):
        return jnp.zeros((num_rnn_layers, batch_size, hidden_size))
    
class ScannedMultiLayerLSTM(nn.Module):
    num_layers: int
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry_LBW_, x_B_):
        ins_BW, resets_B = x_B_
        carry_LBW_ = tuple(
            jnp.where(resets_B[None, :, None], self.initialize_hidden_state(self.num_layers, ins_BW.shape[0], ins_BW.shape[1])[i], carry_LBW_[i])
            for i in range(len(carry_LBW_))
        )
        cs_LBW, hs_LBW = carry_LBW_
        new_cs_LBW, new_hs_LBW = jnp.zeros_like(cs_LBW), jnp.zeros_like(hs_LBW)
        y_BW = ins_BW
        for layer in range(self.num_layers):
            (new_c_BW,  new_h_BW), y_BW = nn.LSTMCell(features=ins_BW.shape[1])(
                (cs_LBW[layer], hs_LBW[layer]), y_BW
            )
            new_cs_LBW = new_cs_LBW.at[layer].set(new_c_BW)
            new_hs_LBW = new_hs_LBW.at[layer].set(new_h_BW)
            
        return (new_cs_LBW, new_hs_LBW), y_BW

    @staticmethod
    def initialize_hidden_state(num_rnn_layers, batch_size, hidden_size):
        single = jnp.zeros((num_rnn_layers, batch_size, hidden_size))
        return (single, single)

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    def setup(self):
        assert self.config["RNN"] in [False, "LSTM", "GRU"]
        NUM_PLAYERS = self.config["ENV_KWARGS"]["num_agents"] if "num_agents" in self.config["ENV_KWARGS"] else 2
        NUM_COLORS = self.config["ENV_KWARGS"]["num_colors"] if "num_colors" in self.config["ENV_KWARGS"] else 5
        NUM_RANKS = self.config["ENV_KWARGS"]["num_ranks"] if "num_ranks" in self.config["ENV_KWARGS"] else 5
        HAND_SIZE = 5 if NUM_PLAYERS < 4 else 4
        self.OBS_PRIVATE_SIZE = (NUM_PLAYERS - 1) * HAND_SIZE * NUM_COLORS * NUM_RANKS

    @nn.compact
    def __call__(self, hidden_state_LBW_, x_TB_):
        obs_TBO, hand_TBY, dones_TB, avail_actions_TBA = x_TB_

        # UNIFORMLY RANDOM POLICY
        if self.config["TRAIN_BELIEF_MODEL"] and not self.config["ACTOR_CRITIC_FOLDER"]:
            action_logits_TBA = jnp.zeros_like(avail_actions_TBA) - 1e10 * (1 - avail_actions_TBA)
            critic_TB = 1.0 * jnp.zeros_like(dones_TB)
            return hidden_state_LBW_, action_logits_TBA, critic_TB

        if self.config["PUBLIC_RNN"]:
            obs_rnn_TB_ = obs_TBO[..., self.OBS_PRIVATE_SIZE:]
        else:
            obs_rnn_TB_ = obs_TBO

        obs_actor_TBO = obs_TBO

        if self.config["CRITIC_CENTRALIZED"]:
            obs_critic_TB_ = jnp.concatenate([hand_TBY, obs_TBO], axis=-1) # Last dimension is Y + O
        else:
            obs_critic_TB_ = obs_TBO

        embedding_obs_actor_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs_actor_TBO))
        embedding_obs_critic_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs_critic_TB_))
        for layer in range(self.config["NUM_RNN_LAYERS"]):
            embedding_obs_actor_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding_obs_actor_TBW))
            embedding_obs_critic_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding_obs_critic_TBW))

        if self.config["RNN"] not in ["LSTM", "GRU"]:
            actor_mean_TBW = embedding_obs_actor_TBW
            critic_mean_TBW = embedding_obs_critic_TBW
        else:
            embedding_obs_rnn_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs_rnn_TB_))

            rnn_in_TB_ = (embedding_obs_rnn_TBW, dones_TB)
            if self.config["RNN"] == "LSTM":
                hidden_state_LBW_, embedding_obs_rnn_TBW = ScannedMultiLayerLSTM(self.config["NUM_RNN_LAYERS"])(hidden_state_LBW_, rnn_in_TB_)
            elif self.config["RNN"] == "GRU":
                hidden_state_LBW_, embedding_obs_rnn_TBW = ScannedMultiLayerGRU(self.config["NUM_RNN_LAYERS"])(hidden_state_LBW_, rnn_in_TB_)

            actor_mean_TBW = embedding_obs_rnn_TBW
            critic_mean_TBW = embedding_obs_rnn_TBW
            if self.config["ACTOR_FF_STREAM"]:
                if self.config["MULTIPLY_PUBLIC_PRIVATE"]:
                    actor_mean_TBW = embedding_obs_actor_TBW * actor_mean_TBW
                else:
                    actor_mean_TBW = jnp.concatenate([embedding_obs_actor_TBW, actor_mean_TBW], axis=-1) # Strictly speaking the last dimension is 2 * B
            if self.config["CRITIC_FF_STREAM"]:
                if self.config["MULTIPLY_PUBLIC_PRIVATE"]:
                    critic_mean_TBW = embedding_obs_critic_TBW * critic_mean_TBW
                else:
                    critic_mean_TBW = jnp.concatenate([embedding_obs_critic_TBW, critic_mean_TBW], axis=-1) # Strictly speaking the last dimension is 2 * B

        for layer in range(self.config["NUM_FF_LAYERS"]):
            actor_mean_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(2), bias_init=constant(0.0))(actor_mean_TBW))
            critic_mean_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(2), bias_init=constant(0.0))(critic_mean_TBW))

        action_logits_TBA = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean_TBW)
        action_logits_TBA = action_logits_TBA - 1e10 * (1 - avail_actions_TBA)

        critic_TB1 = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(critic_mean_TBW)
        critic_TB = critic_TB1.squeeze(-1)

        return hidden_state_LBW_, action_logits_TBA, critic_TB
        
    @staticmethod
    def initialize_hidden_state(rnn, num_rnn_layers, batch_size, hidden_size):
        if rnn == "LSTM":
            initial_hidden_state_LBW_ = ScannedMultiLayerLSTM.initialize_hidden_state(num_rnn_layers, batch_size, hidden_size)
        elif rnn == "GRU":
            initial_hidden_state_LBW_ = ScannedMultiLayerGRU.initialize_hidden_state(num_rnn_layers, batch_size, hidden_size)
        else:
            initial_hidden_state_LBW_ = ScannedMultiLayerGRU.initialize_hidden_state(1, batch_size, 1)
        return initial_hidden_state_LBW_

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    hand: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    env_state: Any
    hidden_state: Any
    encoded_state: jnp.ndarray
    hand_plus_deck: jnp.ndarray
    card_knowledge: jnp.ndarray

def batchify(x_E_: dict, agent_list, num_actors):
    x_PE_ = jnp.stack([x_E_[a] for a in agent_list])
    x_B_ = x_PE_.reshape((num_actors, -1))
    return x_B_

def unbatchify(x_B_: jnp.ndarray, agent_list, num_envs, num_agents):
    x_PE_ = x_B_.reshape((num_agents, num_envs, -1))
    x_E_ = {a: x_PE_[i] for i, a in enumerate(agent_list)}
    return x_E_

class Agent:
    def __init__(self, config, env):
        self.model = ActorCritic(env.action_space(env.agents[0]).n, config=config)
        self.env = env

    def act(self, params, obs_1O, done_1, avail_actions_1A, hidden_state_L1W_):
        hand_1Y = jnp.zeros((obs_1O.shape[0], self.env.hand_size * self.env.num_colors * self.env.num_ranks))
        hidden_state_L1W_, action_logits_11A, _ = self.model.apply(params, hidden_state_L1W_, (obs_1O[None], hand_1Y[None], done_1[None], avail_actions_1A[None])) 
        return hidden_state_L1W_, action_logits_11A.squeeze()
    
    @staticmethod
    def initialize_hidden_state(rnn, num_rnn_layers, batch_size, hidden_size):
        return ActorCritic.initialize_hidden_state(rnn, num_rnn_layers, batch_size, hidden_size)

def get_cross_play_scores(rng, all_agents_params, config, joint_policy_batch_size=1):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    play_action_range = jnp.arange(env.hand_size, 2 * env.hand_size)

    n = len(all_agents_params)
    sp_joint_policy_index_tuples = [tuple([i for _ in range(env.num_agents)]) for i in range(n)]
    xp_joint_policy_index_tuples = list(itertools.permutations(range(n), env.num_agents))
    joint_policy_index_tuples = sp_joint_policy_index_tuples + xp_joint_policy_index_tuples

    num_rollouts = 5000
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, len(joint_policy_index_tuples) * num_rollouts)
    rngs = rngs.reshape((len(joint_policy_index_tuples), num_rollouts, 2))

    agent = Agent(config, env)

    def play_game(rng, agents_params):
        rng, _rng = jax.random.split(rng)
        obs_O_, env_state = env.reset(_rng)
        hidden_states_L1W_ = [agent.initialize_hidden_state(config["RNN"], config["NUM_RNN_LAYERS"], 1, config["LAYER_WIDTH"]) for _ in agents_params]
        agents_params_swapped = agents_params[::-1]
        hidden_states_L1W_swapped = [agent.initialize_hidden_state(config["RNN"], config["NUM_RNN_LAYERS"], 1, config["LAYER_WIDTH"]) for _ in agents_params_swapped]

        cum_kl = 0.0
        kl_count = 1e-10
        cum_reward = 0.0
        done = False
        num_plays, num_neither_known, num_color_known, num_rank_known, num_both_known = 1e-10, 0, 0, 0, 0        

        def cond_fn(val):
            cum_reward, cum_kl, kl_count, rng, done, env_state, obs_O_, hidden_states_L1W_, hidden_states_L1W_swapped, num_plays, num_neither_known, num_color_known, num_rank_known, num_both_known = val
            return jnp.logical_not(done)

        def body_fn(val):
            cum_reward, cum_kl, kl_count, rng, done, env_state, obs_O_, hidden_states_L1W_, hidden_states_L1W_swapped, num_plays, num_neither_known, num_color_known, num_rank_known, num_both_known = val
            avail_actions_A_ = env.get_legal_moves(env_state)
            actions = {}
            for i, agent_id in enumerate(env.agents):
                hidden_states_L1W_[i], action_logits_A = agent.act(
                    agents_params[i], 
                    obs_O_[agent_id][None], 
                    jnp.array([done]), 
                    avail_actions_A_[agent_id][None], 
                    hidden_states_L1W_[i],
                )
                hidden_states_L1W_swapped[i], action_logits_A_swapped = agent.act(
                    agents_params_swapped[i], 
                    obs_O_[agent_id][None], 
                    jnp.array([done]), 
                    avail_actions_A_[agent_id][None], 
                    hidden_states_L1W_swapped[i],
                )
                if config["GREEDY_EVAL"]:
                    action = jnp.argmax(action_logits_A)
                else:
                    rng, _rng = jax.random.split(rng)
                    action = distrax.Categorical(action_logits_A).sample(seed=_rng)
                actions[agent_id] = action

                is_play = 1.0 * jnp.logical_and(
                    play_action_range[0] <= actions[agent_id],
                    play_action_range[-1] >= actions[agent_id]
                )

                pi = distrax.Categorical(action_logits_A)
                pi_swapped = distrax.Categorical(action_logits_A_swapped)
                acting_player_idx = jnp.nonzero(env_state.cur_player_idx, size=1)[0][0]
                cum_kl += (i == 0) * (acting_player_idx == 0) * pi.kl_divergence(pi_swapped)
                #cum_kl += (i == acting_player_idx) * (jnp.argmax(action_logits_A) == jnp.argmax(action_logits_A_swapped))
                kl_count += (i == acting_player_idx)

                def record_card_knowledge():
                    card_position = actions[agent_id] - env.hand_size
                    color, rank = jnp.nonzero(env_state.player_hands[i][card_position], size=1)
                    color, rank = color[0], rank[0]

                    cards_with_same_color_CR = (jnp.arange(env.num_colors)[:, None] == color).astype(jnp.int32)
                    cards_with_same_rank_CR = (jnp.arange(env.num_ranks)[None, :] == rank).astype(jnp.int32)

                    card_knowledge_CR = env_state.card_knowledge[i][card_position].reshape((env.num_colors, env.num_ranks))
                    hand_plus_deck_CR = env_state.player_hands[i].sum(axis=0) + env_state.deck.sum(axis=0)
                    possible_cards_CR = card_knowledge_CR * (0 < hand_plus_deck_CR)
                    color_known = 1.0 * jnp.sum(possible_cards_CR * cards_with_same_color_CR) == jnp.sum(possible_cards_CR)
                    rank_known = 1.0 * jnp.sum(possible_cards_CR * cards_with_same_rank_CR) == jnp.sum(possible_cards_CR)

                    return color_known, rank_known

                def do_nothing():
                    return False, False

                color_known, rank_known = jax.lax.cond(is_play, record_card_knowledge, do_nothing)
                
                num_plays += is_play
                num_neither_known += is_play * (1 - color_known) * (1 - rank_known)
                num_color_known += is_play * color_known * (1 - rank_known)
                num_rank_known += is_play * (1 - color_known) * rank_known
                num_both_known += is_play * color_known * rank_known

            rng, _rng = jax.random.split(rng)
            obs_O_, env_state, reward, dones, _ = env.step(_rng, env_state, actions)

            cum_reward += reward["__all__"]
            done = dones["__all__"]
            return (cum_reward, cum_kl, kl_count, rng, done, env_state, obs_O_, hidden_states_L1W_, hidden_states_L1W_swapped, num_plays, num_neither_known, num_color_known, num_rank_known, num_both_known)

        init_val = (cum_reward, cum_kl, kl_count, rng, done, env_state, obs_O_, hidden_states_L1W_, hidden_states_L1W_swapped, num_plays, num_neither_known, num_color_known, num_rank_known, num_both_known)
        cum_reward, cum_kl, kl_count, _, _, _, _, _, _, num_plays, num_neither_known, num_color_known, num_rank_known, num_both_known = jax.lax.while_loop(cond_fn, body_fn, init_val)
        avg_kl = cum_kl / kl_count
        percent_neither_known = num_neither_known / num_plays 
        percent_color_known = num_color_known / num_plays
        percent_rank_known = num_rank_known / num_plays 
        percent_both_known = num_both_known / num_plays

        return cum_reward, avg_kl, (percent_neither_known, percent_color_known, percent_rank_known, percent_both_known)

    def evaluate_joint_policy(rngs, agents_params):
        return jax.vmap(play_game, (0, None))(rngs, agents_params)

    @jax.jit 
    def evaluate_joint_policies_batch(rngs, joint_policies):
        return jax.vmap(evaluate_joint_policy, (0, 0))(rngs, joint_policies)

    # --- Batched loop over joint policies ---
    J = rngs.shape[0]
    scores_batch_outs = []
    kl_batch_outs = []
    percent_neither_known_batch_outs = []
    percent_color_known_batch_outs = []
    percent_rank_known_batch_outs = []
    percent_both_known_batch_outs = []

    for start in range(0, J, joint_policy_batch_size):
        time_before_batch = time.time()
        stop = min(start + joint_policy_batch_size, J)
        rngs_batch = rngs[start:stop]
        joint_policy_index_tuples_batch = joint_policy_index_tuples[start:stop]
        joint_policies_list_batch = [[all_agents_params[i] for i in index_tuple] for index_tuple in joint_policy_index_tuples_batch]
        joint_policies_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *joint_policies_list_batch)
        cum_reward, cum_kl, (percent_neither_known, percent_color_known, percent_rank_known, percent_both_known) = evaluate_joint_policies_batch(rngs_batch, joint_policies_batch)

        scores_batch_outs.append(cum_reward)
        kl_batch_outs.append(cum_kl)
        percent_neither_known_batch_outs.append(percent_neither_known)
        percent_color_known_batch_outs.append(percent_color_known)
        percent_rank_known_batch_outs.append(percent_rank_known)
        percent_both_known_batch_outs.append(percent_both_known)

        time_after_batch = time.time()
        time_per_joint_policy = (time_after_batch - time_before_batch) / joint_policy_batch_size
        print("Time per joint_policy: ", time_per_joint_policy)

    scores = jnp.concatenate(scores_batch_outs, axis=0)
    kl = jnp.concatenate(kl_batch_outs, axis=0)
    percent_neither_known = jnp.concatenate(percent_neither_known_batch_outs, axis=0)
    percent_color_known = jnp.concatenate(percent_color_known_batch_outs, axis=0)
    percent_rank_known = jnp.concatenate(percent_rank_known_batch_outs, axis=0)
    percent_both_known = jnp.concatenate(percent_both_known_batch_outs, axis=0)

    # Split back into SP / XP and average over rollouts
    sp_len = len(sp_joint_policy_index_tuples)
    sp_scores = scores[:sp_len].mean(axis=-1)
    xp_scores = scores[sp_len:].mean(axis=-1)

    sp_kl = kl[:sp_len].mean(axis=-1)
    xp_kl = kl[sp_len:].mean(axis=-1)
    
    sp_percent_neither_known = percent_neither_known[:sp_len].mean(axis=-1)
    xp_percent_neither_known = percent_neither_known[sp_len:].mean(axis=-1)

    sp_percent_color_known = percent_color_known[:sp_len].mean(axis=-1)
    xp_percent_color_known = percent_color_known[sp_len:].mean(axis=-1)

    sp_percent_rank_known = percent_rank_known[:sp_len].mean(axis=-1)
    xp_percent_rank_known = percent_rank_known[sp_len:].mean(axis=-1)

    sp_percent_both_known = percent_both_known[:sp_len].mean(axis=-1)
    xp_percent_both_known = percent_both_known[sp_len:].mean(axis=-1)

    return sp_scores, xp_scores, sp_kl, xp_kl, (sp_percent_neither_known, sp_percent_color_known, sp_percent_rank_known, sp_percent_both_known), (xp_percent_neither_known, xp_percent_color_known, xp_percent_rank_known, xp_percent_both_known)

def play_game_render(rng, agents_params, config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    agent = Agent(config, env)
    rng, _rng = jax.random.split(rng)
    obs_O_, env_state = env.reset(_rng)
    hidden_states_L1W_ = [agent.initialize_hidden_state(config["RNN"], config["NUM_RNN_LAYERS"], 1, config["LAYER_WIDTH"]) for _ in agents_params]
    cum_reward = 0.0
    done = False
    turn = 0
    
    while not done:
        env.render(env_state)
        avail_actions_A_ = env.get_legal_moves(env_state)
        actions = {}
        for i, agent_id in enumerate(env.agents):
            hidden_states_L1W_[i], actions[agent_id] = agent.act(
                agents_params[i], obs_O_[agent_id][None], jnp.array([done]), avail_actions_A_[agent_id][None], hidden_states_L1W_[i]
            )
            actions[agent_id] = actions[agent_id][0]
            if i == 0 and turn % len(env.agents) == 0:
                print("AI action: ", actions[agent_id])
                actions[agent_id] = int(input("Enter your action: "))

        rng, _rng = jax.random.split(rng)
        obs_O_, env_state, reward, dones, _ = env.step(_rng, env_state, actions)

        cum_reward += reward["__all__"]
        done = dones["__all__"]
        turn += 1

def plot_xp_matrix(sp_scores, xp_scores, labels=None, save_path=None):
    d = len(sp_scores)
    assert len(xp_scores) == d*(d-1), "xp_scores must have length d(d-1)"

    sp_mean = jnp.mean(sp_scores)
    sp_std = jnp.std(sp_scores)
    xp_mean = jnp.mean(xp_scores)
    xp_std = jnp.std(xp_scores)
    
    # Build the matrix
    M = np.zeros((d, d))
    off_idx = 0
    for i in range(d):
        for j in range(d):
            if i == j:
                M[i, j] = sp_scores[i]
            else:
                M[i, j] = xp_scores[off_idx]
                off_idx += 1

    M = M[::-1]
    
    # Custom red→green colormap
    cmap = LinearSegmentedColormap.from_list("red_green", ["red", "yellow", "green"])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Write numbers inside the matrix
    for i in range(d):
        for j in range(d):
            ax.text(j, i, f"{M[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=5)
    
    ax.set_xticks(1.5 + 4 * np.arange(10))
    ax.set_yticks(1.5 + 4 * np.arange(10))
    labels = np.round(np.arange(0.01, 0.11, 0.01), 2)
    ax.set_xticklabels(labels) #, rotation=45, ha="right")
    ax.set_yticklabels(labels[::-1])
    
    plt.tight_layout()
    
    # Save figure if path is given
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    plt.show()

def make_train(config, run_name, wandb_run_name):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["TRAIN_BELIEF_MODEL"]:
        assert config["TRAIN_POLICY"] == False
        config["OBL"] == False
        config["ACTING_PLAYER_ONLY"] = False
        config["BELIEF_MODEL_FOLDER"] = False
        config["NUM_STEPS"] = config["BM_NUM_STEPS"]
        config["TOTAL_TIMESTEPS"] = config["BM_TOTAL_TIMESTEPS"]

    if config["OBL"]:
        config["PUBLIC_RNN"] = True
        if config["ACTING_PLAYER_ONLY"]:
            assert config["k"] % env.num_agents == 0
            config["GAMMA"] = 1
            config["GAE_LAMBDA"] = 1
    else:
        config["ACTING_PLAYER_ONLY"] = False # Necessary since GAE always uses values from the perspective of both acting and non-acting player

    if config["PUBLIC_RNN"]:
        config["ACTOR_FF_STREAM"] = True
        config["CRITIC_FF_STREAM"] = True

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    config["NUM_OUTER_UPDATES"] = config["NUM_UPDATES"] // config["LOG_INTERVAL"]
    config["NUM_UPDATES_PER_INNER_LOOP"] = config["NUM_UPDATES"] // config["NUM_OUTER_UPDATES"]
    config["NUM_GPUS"] = jax.device_count()

    if config["BELIEF_MODEL_FOLDER"] == False:
        belief_model_params_stacked = jnp.zeros(config["NUM_GPUS"])
    else:
        belief_model_filenames = sorted([f for f in os.listdir(config["BELIEF_MODEL_FOLDER"]) if f.endswith(".safetensors")])
        belief_model_params_list = [unflatten_dict(load_file(os.path.join(config["BELIEF_MODEL_FOLDER"], f)), sep=",") for f in belief_model_filenames]
        belief_model_params_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *belief_model_params_list[:config["NUM_GPUS"]])

    if config["ACTOR_CRITIC_FOLDER"] == False:
        actor_critic_params_stacked = jnp.zeros(config["NUM_GPUS"])
    else:
        actor_critic_filenames = sorted([f for f in os.listdir(config["ACTOR_CRITIC_FOLDER"]) if f.endswith(".safetensors")])
        actor_critic_params_list = [unflatten_dict(load_file(os.path.join(config["ACTOR_CRITIC_FOLDER"], f)), sep=",") for f in actor_critic_filenames]
        actor_critic_params_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *actor_critic_params_list[:config["NUM_GPUS"]])

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def _get_init_update_state(rng, actor_critic_params, belief_model_params):
        # INIT ACTOR-CRITIC NETWORK
        ac_network = ActorCritic(env.action_space(env.agents[0]).n, config=config)

        if config["ACTOR_CRITIC_FOLDER"] == False:
            rng, _rng = jax.random.split(rng)
            init_x = (
                jnp.zeros((1, config["NUM_ACTORS"], env.observation_space(env.agents[0]).shape)), # obs
                jnp.zeros((1, config["NUM_ACTORS"], env.hand_size * env.num_colors * env.num_ranks)), # hand
                jnp.zeros((1, config["NUM_ACTORS"])), # done
                jnp.zeros((1, config["NUM_ACTORS"], env.action_space(env.agents[0]).n)) # avail_actions
            )
            init_hidden_state = ActorCritic.initialize_hidden_state(config["RNN"], config["NUM_RNN_LAYERS"], config["NUM_ACTORS"], config["LAYER_WIDTH"])
            ac_network_params = ac_network.init(_rng, init_hidden_state, init_x)
        else:
            ac_network_params = actor_critic_params

        if config["ANNEAL_LR"]:
            ac_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
        else:
            ac_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

        train_state = TrainState.create(
            apply_fn=ac_network.apply,
            params=ac_network_params,
            tx=ac_tx,
        )

        # INIT BELIEF MODEL NETWORK
        count_based = config["TRAIN_POLICY"] and not config["BELIEF_MODEL_FOLDER"]
        belief_model = BeliefModel(
            hidden_size=config["LAYER_WIDTH"],
            num_colors=env.num_colors, 
            num_ranks=env.num_ranks,
            hand_size=env.hand_size,
            num_rnn_layers=config["NUM_RNN_LAYERS"],
            num_ff_layers_encoder=config["NUM_FF_LAYERS_ENCODER"],
            num_ff_layers_decoder=config["NUM_FF_LAYERS_DECODER"],
            count_based=count_based,
        )

        if config["BELIEF_MODEL_FOLDER"] == False:
            init_x = (
                jnp.zeros((1, config["NUM_ACTORS"], env.observation_space(env.agents[0]).shape)), # obs
                jnp.zeros((1, config["NUM_ACTORS"])), # done
                jnp.zeros((1, config["NUM_ACTORS"], env.hand_size * env.num_colors * env.num_ranks)), # hand
                jnp.zeros((1, config["NUM_ACTORS"], env.num_colors * env.num_ranks)), # hand_plus_deck
                jnp.zeros((1, config["NUM_ACTORS"], env.hand_size, env.num_colors * env.num_ranks)), # card_knowledge
            )
            init_hidden = LSTMEncoder.initialize_hidden_state(config["NUM_RNN_LAYERS"], config["NUM_ACTORS"], config["LAYER_WIDTH"])
            rng, _rng = jax.random.split(rng)
            bm_params = belief_model.init(_rng, (_rng, init_hidden), init_x)
        else:
            bm_params = belief_model_params
        
        if config["ANNEAL_LR"]:
            bm_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
        else:
            bm_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=config["BM_LR"], eps=1e-5))
            
        bm_train_state = TrainState.create(
            apply_fn=belief_model.apply,
            params=bm_params,
            tx=bm_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        rng_E_ = jax.random.split(_rng, config["NUM_ENVS"])
        obs_EO_, env_state_E_ = jax.vmap(env.reset, in_axes=(0,))(rng_E_)
        done_B = jnp.ones((config["NUM_ACTORS"]), dtype=bool)
        init_hidden_state_LBW_ = ActorCritic.initialize_hidden_state(config["RNN"], config["NUM_RNN_LAYERS"], config["NUM_ACTORS"], config["LAYER_WIDTH"])
        bm_init_hidden_state_LBW_ = LSTMEncoder.initialize_hidden_state(config["NUM_RNN_LAYERS"], config["NUM_ACTORS"], config["LAYER_WIDTH"])

        runner_state = (train_state, bm_train_state, env_state_E_, obs_EO_, done_B, init_hidden_state_LBW_, bm_init_hidden_state_LBW_, rng)

        return (runner_state, 0)

    @jax.jit
    def update_agents(update_state, process_id):

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            init_hidden_state_LBW_, bm_init_hidden_state_LBW_ = runner_state[-3], runner_state[-2]
            def _env_step(runner_state, predetermined_action_B):
                train_state, bm_train_state, env_state_E_, obs_EO_, done_B, hidden_state_LBW_, bm_hidden_state_LBW_, rng = runner_state # Could make obs and done either both batches or both dicts

                # SELECT ACTION
                avail_actions_EA_ = jax.vmap(env.get_legal_moves)(env_state_E_.env_state)
                avail_actions_BA = jax.lax.stop_gradient(batchify(avail_actions_EA_, env.agents, config["NUM_ACTORS"]))
                obs_BO = batchify(obs_EO_, env.agents, config["NUM_ACTORS"])
                hand_EPHCR = env_state_E_.env_state.player_hands
                hand_BHX = hand_EPHCR.swapaxes(0, 1).reshape((config["NUM_ACTORS"], hand_EPHCR.shape[2], -1))
                deck_EDCR = env_state_E_.env_state.deck
                deck_BDX = jnp.stack([deck_EDCR] * env.num_agents).reshape((config["NUM_ACTORS"], deck_EDCR.shape[1], -1))
                hand_plus_deck_BX = hand_BHX.sum(axis=1) + deck_BDX.sum(axis=1)
                hand_BY = hand_BHX.reshape((config["NUM_ACTORS"], -1))
                card_knowledge_EPHX = env_state_E_.env_state.card_knowledge
                card_knowledge_BHX = card_knowledge_EPHX.swapaxes(0, 1).reshape((config["NUM_ACTORS"], ) + card_knowledge_EPHX.shape[2:])

                ac_in_1B_ = (obs_BO[None], hand_BY[None], done_B[None], avail_actions_BA[None])
                next_hidden_state_LBW_, action_logits_1BA, value_1B = train_state.apply_fn(train_state.params, hidden_state_LBW_, ac_in_1B_)
                action_logits_BA = action_logits_1BA.squeeze()
                value_B = value_1B.squeeze()
                pi_BA_ = distrax.Categorical(action_logits_BA)
                rng, _rng = jax.random.split(rng)
                sampled_action_B = pi_BA_.sample(seed=_rng)
                action_B = sampled_action_B * (predetermined_action_B == -1) + predetermined_action_B * (predetermined_action_B != -1)
                log_prob_B = pi_BA_.log_prob(action_B)
                env_act_E1_ = unbatchify(action_B, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act_E_ = jax.tree.map(lambda x: x.squeeze(), env_act_E1_)

                # Encode history TODO DURING FICTITIOUS ROLLOUTS THIS DOESN'T NEED TO BE COMPUTED!!!
                next_bm_hidden_state_LBW_, encoded_state_BW = bm_train_state.apply_fn(
                    bm_train_state.params, 
                    bm_hidden_state_LBW_, obs_BO, done_B,
                    method=BeliefModel.encode
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_E_ = jax.random.split(_rng, config["NUM_ENVS"])
                next_obs_EO_, next_env_state_E_, reward_E_, next_done_E_, info_EP_ = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_E_, env_state_E_, env_act_E_
                )
                reward_B = batchify(reward_E_, env.agents, config["NUM_ACTORS"]).squeeze()
                next_done_B = batchify(next_done_E_, env.agents, config["NUM_ACTORS"]).squeeze()
                info_B_ = jax.tree.map(lambda x: x.swapaxes(0, 1).reshape((config["NUM_ACTORS"])), info_EP_)
                transition_B_ = Transition(
                    done_B,
                    action_B,
                    value_B,
                    reward_B,
                    log_prob_B,
                    obs_BO,
                    hand_BY,
                    info_B_,
                    avail_actions_BA,
                    env_state_E_, # Technically doesn't have leading dimension B, but E. Doesn't matter though, as not used in gradient updates.
                    hidden_state_LBW_,
                    encoded_state_BW,
                    hand_plus_deck_BX,
                    card_knowledge_BHX,
                )
                runner_state = (train_state, bm_train_state, next_env_state_E_, next_obs_EO_, next_done_B, next_hidden_state_LBW_, next_bm_hidden_state_LBW_, rng)
                return runner_state, transition_B_

            predetermined_actions_TB = - jnp.ones((config["NUM_STEPS"], config["NUM_ACTORS"]), dtype=int)
            runner_state, traj_batch_TB_ = jax.lax.scan(
                _env_step, runner_state, predetermined_actions_TB, config["NUM_STEPS"]
            )

            train_state, bm_train_state, last_env_state_E_, last_obs_EO_, last_done_B, last_hidden_state_LBW_, last_bm_hidden_state_LBW_, rng = runner_state

            # COMPUTE TD(LAMBDA) TARGETS
            def _compute_targets(last_env_state_E_, last_obs_EO_, last_done_B, last_hidden_state_LBW_, traj_batch_TB_):
                last_obs_BO = batchify(last_obs_EO_, env.agents, config["NUM_ACTORS"])
                last_hand_EPHCR = last_env_state_E_.env_state.player_hands
                last_hand_BY = last_hand_EPHCR.swapaxes(0, 1).reshape((config["NUM_ACTORS"],-1))
                avail_actions_EA_ = avail_actions = jax.vmap(env.get_legal_moves)(last_env_state_E_.env_state)
                avail_actions_BA = jax.lax.stop_gradient(batchify(avail_actions, env.agents, config["NUM_ACTORS"]))
                ac_in_1B_ = (last_obs_BO[None], last_hand_BY[None], last_done_B[None], avail_actions_BA[None])
                _, _, last_value_1B = train_state.apply_fn(train_state.params, last_hidden_state_LBW_, ac_in_1B_)
                last_value_B = last_value_1B.squeeze()

                def _compute_targets_from_gae(traj_batch_TB_, last_value_B, last_done_B):
                    def _get_gae(gae_and_next_value_B_, transition_B_):
                        gae_B, next_value_B, next_done_B = gae_and_next_value_B_
                        done_B, value_B, reward_B = (
                            transition_B_.done,
                            transition_B_.value,
                            transition_B_.reward,
                        )
                        delta_B  = reward_B + config["GAMMA"] * next_value_B * (1 - next_done_B) - value_B
                        gae_B  = delta_B  + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done_B) * gae_B
                        return (gae_B, value_B, done_B), gae_B

                    init_B_ = (jnp.zeros_like(last_value_B), last_value_B, last_done_B)
                    _, gae_TB = jax.lax.scan(
                        _get_gae,
                        init_B_,
                        traj_batch_TB_,
                        reverse=True,
                        unroll=16,
                    )
                    return gae_TB + traj_batch_TB_.value

                targets_TB = _compute_targets_from_gae(traj_batch_TB_, last_value_B, last_done_B)
                return targets_TB

            # COLLECT FICTITIOUS ROLLOUTS
            def _get_fict_traj_batch(rng, traj_batch_TB_):
                def _sample_fict_state_and_obs(traj_batch_B_, rng):
                    acting_player_idx_E = jax.vmap(lambda x: jnp.nonzero(x, size=1)[0][0])(traj_batch_B_.env_state.env_state.cur_player_idx)
                    
                    fict_env_state_E_, bm_metrics_E_ = sample_hand_from_belief_learned(
                        bm_train_state,
                        rng,
                        traj_batch_B_.env_state.env_state, 
                        traj_batch_B_.encoded_state, 
                        traj_batch_B_.obs, 
                        acting_player_idx_E,
                    )
                    fict_env_state_E_ = traj_batch_B_.env_state.replace(env_state=fict_env_state_E_)

                    fict_hands_EPHCR = fict_env_state_E_.env_state.player_hands
                    fict_hands_EPY = fict_hands_EPHCR.reshape(fict_hands_EPHCR.shape[:2] + (-1,))

                    def other_hands(aidx):
                        hands_from_self_EPY = jnp.roll(fict_hands_EPY, -aidx, axis=1)
                        other_hands_EZ = hands_from_self_EPY[:, 1:].reshape((hands_from_self_EPY.shape[0], -1))
                        return other_hands_EZ

                    other_hands_PEZ = jnp.stack([other_hands(aidx) for aidx in range(env.num_agents)])
                    obs_PEO = traj_batch_B_.obs.reshape(other_hands_PEZ.shape[:2] + (-1,))
                    fict_obs_PEO = obs_PEO.at[:, :, :other_hands_PEZ.shape[-1]].set(other_hands_PEZ)
                    fict_obs_BO = fict_obs_PEO.reshape((config["NUM_ACTORS"], -1))
                    fict_obs_EO_ = unbatchify(fict_obs_PEO, env.agents, config["NUM_ENVS"], env.num_agents)

                    return fict_env_state_E_, fict_obs_EO_, bm_metrics_E_

                rng, _rng = jax.random.split(rng)
                rng_T_ = jax.random.split(_rng, config["NUM_STEPS"])
                fict_env_state_TE_, fict_obs_TEO_, bm_metrics_TE_ = jax.vmap(_sample_fict_state_and_obs, (0, 0))(traj_batch_TB_, rng_T_)

                done_TB, hidden_state_TLBW_, action_TB = traj_batch_TB_.done, traj_batch_TB_.hidden_state, traj_batch_TB_.action

                rng, _rng = jax.random.split(rng)
                rng_T_ = jax.random.split(_rng, config["NUM_STEPS"])
                bm_hidden_state_TLBW_ = jax.tree.map(lambda x: jnp.zeros((config["NUM_STEPS"],) + x.shape), last_bm_hidden_state_LBW_) # Dummy, as it doesn't matter anymore after the fictitious state is already sampled.
                fict_runner_state_T_ = (fict_env_state_TE_, fict_obs_TEO_, done_TB, hidden_state_TLBW_, bm_hidden_state_TLBW_, rng_T_)

                predetermined_actions_TKB = - jnp.ones((config["NUM_STEPS"], config["k"], config["NUM_ACTORS"]), dtype=int)
                predetermined_actions_TKB = predetermined_actions_TKB.at[:, 0].set(action_TB)

                def _scanned_env_step(train_state, bm_train_state, fict_runner_state, predetermined_actions_KB):
                    fict_runner_state = (train_state, bm_train_state) + fict_runner_state
                    fict_runner_state, fict_traj_batch_KB_ = jax.lax.scan(
                        _env_step, fict_runner_state, predetermined_actions_KB, config["k"]
                    )
                    return fict_runner_state[2:], fict_traj_batch_KB_
                fict_runner_state_T_, fict_traj_batch_TKB_ = jax.vmap(_scanned_env_step, (None, None, 0, 0))(train_state, bm_train_state, fict_runner_state_T_, predetermined_actions_TKB)
                return fict_runner_state_T_, fict_traj_batch_TKB_, bm_metrics_TE_

            if config["TRAIN_POLICY"] and config["OBL"]:
                fict_runner_state_T_, fict_traj_batch_TKB_, bm_metrics_TE_ = _get_fict_traj_batch(rng, traj_batch_TB_)

                fict_last_env_state_TE_, fict_last_obs_TEO_, fict_last_done_TB, fict_last_hidden_state_TLBW, _, _ = fict_runner_state_T_

                fict_targets_TKB = jax.vmap(_compute_targets, (0, 0, 0, 0, 0))(
                    fict_last_env_state_TE_, fict_last_obs_TEO_, fict_last_done_TB, fict_last_hidden_state_TLBW, fict_traj_batch_TKB_
                )
                traj_batch_TB_ = jax.tree.map(lambda x: x[:, 0], fict_traj_batch_TKB_)
                targets_TB = fict_targets_TKB[:, 0]
                gae_TB = targets_TB - traj_batch_TB_.value

            elif config["TRAIN_POLICY"] and not config["OBL"]:
                targets_TB = _compute_targets(
                    last_env_state_E_, last_obs_EO_, last_done_B, last_hidden_state_LBW_, traj_batch_TB_
                )
                gae_TB = targets_TB - traj_batch_TB_.value

                bm_metrics_TE_ = {
                    "valid_percentage_hands": jnp.ones(1),
                    "entropy_legal": jnp.zeros(1),
                    "entropy_count": jnp.zeros(1),
                }

            elif not config["TRAIN_POLICY"]:
                targets_TB = jnp.zeros_like(traj_batch_TB_.value)
                gae_TB = jnp.zeros_like(traj_batch_TB_.value)

                bm_metrics_TE_ = {
                    "valid_percentage_hands": jnp.ones(1),
                    "entropy_legal": jnp.zeros(1),
                    "entropy_count": jnp.zeros(1),
                }
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    train_state, bm_train_state = train_states
                    init_hidden_state_LMW_, bm_init_hidden_state_LMW_, traj_batch_TM_, gae_TM, targets_TM = batch_info

                    def _loss_fn(params):
                        actor_mask_TM = (1 < traj_batch_TM_.avail_actions.sum(axis=-1))
                        critic_mask_TM = 1 + jnp.zeros_like(actor_mask_TM)
                        if config["ACTING_PLAYER_ONLY"]:
                            critic_mask_TM *= actor_mask_TM

                        # RERUN NETWORK
                        _, action_logits_TMA, value_TM = train_state.apply_fn(
                            params, 
                            init_hidden_state_LMW_,
                            (traj_batch_TM_.obs, traj_batch_TM_.hand, traj_batch_TM_.done, traj_batch_TM_.avail_actions),
                        )
                        pi_TMA_ = distrax.Categorical(action_logits_TMA)
                        log_prob_TM = pi_TMA_.log_prob(traj_batch_TM_.action)

                        # CALCULATE CRITIC LOSS
                        value_pred_clipped_TM = traj_batch_TM_.value + (
                                value_TM - traj_batch_TM_.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_TM = jnp.square(value_TM - targets_TM)
                        value_losses_clipped_TM = jnp.square(value_pred_clipped_TM - targets_TM)
                        value_losses_TM = jnp.maximum(value_losses_TM, value_losses_clipped_TM)
                        value_loss = 0.5 * (critic_mask_TM * value_losses_TM).sum()/critic_mask_TM.sum()

                        # CALCULATE ACTOR LOSS
                        logratio_TM = log_prob_TM - traj_batch_TM_.log_prob
                        ratio_TM = jnp.exp(logratio_TM)
                        gae_mean = jnp.sum(actor_mask_TM * gae_TM) / actor_mask_TM.sum()
                        gae_std = jnp.sqrt(
                            jnp.sum(actor_mask_TM * (gae_TM - gae_mean)**2) / actor_mask_TM.sum()
                        )
                        gae_normalized_TM = (gae_TM - gae_mean) / (gae_std + 1e-10)
                        actor_losses_TM = ratio_TM * gae_normalized_TM
                        actor_losses_clipped_TM = (
                                jnp.clip(
                                    ratio_TM,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae_normalized_TM
                        )
                        actor_losses_TM = -jnp.minimum(actor_losses_TM, actor_losses_clipped_TM)
                        actor_loss = (actor_mask_TM * actor_losses_TM).sum() / actor_mask_TM.sum()
                        
                        entropy = (actor_mask_TM * pi_TMA_.entropy()).sum() / actor_mask_TM.sum()

                        total_loss = (
                                actor_loss
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                        )

                        # EXTRA DIAGNOSTICS
                        clip_frac = (actor_mask_TM * (jnp.abs(ratio_TM - 1) > config["CLIP_EPS"])).sum() / actor_mask_TM.sum()
                        ratio = ratio_TM.mean()

                        return total_loss, (value_loss, actor_loss, entropy, clip_frac, ratio)

                    if config["TRAIN_POLICY"]:
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params,
                        )
                        train_state = train_state.apply_gradients(grads=grads)
                    else:
                        total_loss = (0, (0, 0, 0, 0, 0, 0))

                    def _bm_loss_fn(bm_params):
                        (_, _), (logits_new_card_count_TMHX, logits_new_card_legal_TMHX) = bm_train_state.apply_fn(
                            bm_params, 
                            (_rng, bm_init_hidden_state_LMW_), 
                            (traj_batch_TM_.obs, traj_batch_TM_.done, traj_batch_TM_.hand, traj_batch_TM_.hand_plus_deck, traj_batch_TM_.card_knowledge),
                        )

                        hand_TMHX = traj_batch_TM_.hand.reshape(traj_batch_TM_.hand.shape[:2] + (env.hand_size, -1))
                        loss_count = jnp.mean(optax.losses.softmax_cross_entropy(logits_new_card_count_TMHX, hand_TMHX))
                        loss_legal = jnp.mean(optax.losses.softmax_cross_entropy(logits_new_card_legal_TMHX, hand_TMHX))

                        entropy_count = jnp.mean(distrax.Categorical(logits_new_card_count_TMHX).entropy())
                        entropy_legal = jnp.mean(distrax.Categorical(logits_new_card_legal_TMHX).entropy())
                        return loss_legal, (entropy_legal, loss_count)

                    if config["TRAIN_BELIEF_MODEL"]:
                        bm_grad_fn = jax.value_and_grad(_bm_loss_fn, has_aux=True)
                        bm_total_loss, bm_grads = bm_grad_fn(
                            bm_train_state.params,
                        )
                        bm_train_state = bm_train_state.apply_gradients(grads=bm_grads)
                    else:
                        bm_total_loss = (0, (0, 0))

                    return (train_state, bm_train_state), (total_loss, bm_total_loss)

                train_state, bm_train_state, init_hidden_state_LBW_, bm_init_hidden_state_LBW_, traj_batch_TB_, gae_TB, targets_TB, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (init_hidden_state_LBW_, bm_init_hidden_state_LBW_, traj_batch_TB_, gae_TB, targets_TB)

                # Shuffle along the batch axis
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        1, 0,
                    ),
                    shuffled_batch,
                )

                (train_state, bm_train_state), (total_loss, bm_total_loss) = jax.lax.scan(
                    _update_minbatch, (train_state, bm_train_state), minibatches
                )
                update_state = (train_state, bm_train_state, init_hidden_state_LBW_, bm_init_hidden_state_LBW_, traj_batch_TB_, gae_TB, targets_TB, rng)
                return update_state, (total_loss, bm_total_loss)

            update_state = (train_state, bm_train_state, init_hidden_state_LBW_, bm_init_hidden_state_LBW_, traj_batch_TB_, gae_TB, targets_TB, rng)
            update_state, (loss_info, bm_loss_info) = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state, bm_train_state = update_state[0], update_state[1]
            metric_TB_ = traj_batch_TB_.info
            metric = jax.tree.map(lambda x: x[-1].mean(), metric_TB_)
            
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            bm_loss_info = jax.tree.map(lambda x: x.mean(), bm_loss_info)
            bm_metrics = jax.tree.map(lambda x: x.mean(), bm_metrics_TE_)

            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "loss_actor": loss_info[1][1],
                "entropy": loss_info[1][2],
                "clip_frac": loss_info[1][3],
                "ratio": loss_info[1][4],
                "valid_percentage_hands": bm_metrics["valid_percentage_hands"],
                "entropy_learned_BM": bm_metrics["entropy_legal"],
                "entropy_count_BM": bm_metrics["entropy_count"],
                "loss_belief_model": bm_loss_info[0],
                "entropy_belief_model": bm_loss_info[1][0],
                "loss_count": bm_loss_info[1][1],
            }

            def callback(metric, process_id, update_steps):
                metric["loss"] = {key + f"_{process_id}": metric["loss"][key] for key in metric["loss"]}
                dict_to_log = {
                    f"SP_{process_id}": metric["returned_episode_returns"],
                    f"UpdateStep": update_steps,
                    **metric["loss"],
                }
                wandb.log(dict_to_log)

            jax.experimental.io_callback(callback, None, metric, process_id, update_steps)
            rng = update_state[-1]
            update_steps = update_steps + 1

            runner_state = (train_state, bm_train_state, last_env_state_E_, last_obs_EO_, last_done_B, last_hidden_state_LBW_, last_bm_hidden_state_LBW_, rng)

            update_runner_state = (runner_state, update_steps)
            return update_runner_state, None

        update_state, metric = jax.lax.scan(_update_step, update_state, None, config["NUM_UPDATES_PER_INNER_LOOP"])
        return update_state

    def train(rng):
        if not config["TRAIN_POLICY"] and not config["TRAIN_BELIEF_MODEL"]:
            assert config["ACTOR_CRITIC_FOLDER"] != False
            if True:
                rng, _rng = jax.random.split(rng)
                sp_scores, xp_scores, sp_kl, xp_kl, sp_play_card_knowledge, xp_play_card_knowledge = get_cross_play_scores(_rng, actor_critic_params_list, config)
                np.save(config["ACTOR_CRITIC_FOLDER"] + "/" + "SP_kl.npy", sp_kl)
                np.save(config["ACTOR_CRITIC_FOLDER"] + "/" + "XP_kl.npy", xp_kl)
                np.save(config["ACTOR_CRITIC_FOLDER"] + "/" + "SP_scores.npy", sp_scores)
                np.save(config["ACTOR_CRITIC_FOLDER"] + "/" + "XP_scores.npy", xp_scores)
                np.save(config["ACTOR_CRITIC_FOLDER"] + "/" + "SP_play_card_knowledge.npy", sp_play_card_knowledge)
                np.save(config["ACTOR_CRITIC_FOLDER"] + "/" + "XP_play_card_knowledge.npy", xp_play_card_knowledge)

            sp_scores = np.load(config["ACTOR_CRITIC_FOLDER"] + "/" + "SP_scores.npy")
            xp_scores = np.load(config["ACTOR_CRITIC_FOLDER"] + "/" + "XP_scores.npy")
            sp_kl = np.load(config["ACTOR_CRITIC_FOLDER"] + "/" + "SP_kl.npy")
            xp_kl = np.load(config["ACTOR_CRITIC_FOLDER"] + "/" + "XP_kl.npy")
            plot_xp_matrix(sp_scores, xp_scores, actor_critic_filenames, config["ACTOR_CRITIC_FOLDER"] + "/" + "XP_Matrix2")
            plot_xp_matrix(sp_kl, xp_kl, actor_critic_filenames, config["ACTOR_CRITIC_FOLDER"] + "/" + "KL_Matrix3")

            rng, _rng = jax.random.split(rng)
            #play_game_render(_rng, actor_critic_params_list[-2:], config)
            return None
        
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, config["NUM_GPUS"])
        update_states = jax.pmap(_get_init_update_state, axis_name="devices")(_rng, actor_critic_params_stacked, belief_model_params_stacked)
    
        pmapped_update = jax.pmap(update_agents, axis_name="devices")

        wandb.log({
                "Avg SP": 0,
                "Min SP": 0,
                "Max SP": 0,
                "Avg XP": 0,
                "UpdateStep": 0
        })
        last_call_time = time.time()

        print("Number of outer updates: ", config["NUM_OUTER_UPDATES"])
        for i in range(config["NUM_OUTER_UPDATES"]):
            print('\n')
            print("Outer Update Step:", i+1)

            before_update = time.time()
            update_states = pmapped_update(update_states, jnp.arange(config["NUM_GPUS"]))
            jax.block_until_ready(update_states)
            now = time.time()
            print("Time for Update:", now-before_update)

            # GET ALL WEIGHTS ON ONE GPU
            update_states_one_gpu = jax.tree.map(lambda x: jax.device_put(x, jax.devices()[0]), update_states)
            train_states = update_states_one_gpu[0][0]
            actor_critic_params = [jax.tree.map(lambda x: x[j], train_states).params for j in range(config["NUM_GPUS"])]
            bm_train_states = update_states_one_gpu[0][1] # TODO
            update_steps = update_states_one_gpu[1][0]

            # Create WANDB DICT_TO_LOG
            dict_to_log = {}

            # COMPUTE SP AND XP SCORES
            if config["XP_LOGGING"] and not config["TRAIN_BELIEF_MODEL"]:
                before_xp = time.time()
                rng, _rng = jax.random.split(rng)
                sp_scores, xp_scores, sp_kl, xp_kl, sp_play_card_knowledge, xp_play_card_knowledge = get_cross_play_scores(_rng, actor_critic_params, config)
                avg_sp_score, avg_xp_score = sp_scores.mean(), xp_scores.mean()
                min_sp_score, max_sp_score = sp_scores.min(), sp_scores.max()
                avg_sp_kl, avg_xp_kl = sp_kl.mean(), xp_kl.mean()

                dict_to_log["Avg SP"] = avg_sp_score
                dict_to_log["Min SP"] = min_sp_score
                dict_to_log["Max SP"] = max_sp_score
                dict_to_log["Avg XP"] = avg_xp_score
                dict_to_log["Avg KL SP"] = avg_sp_kl
                dict_to_log["Avg KL XP"] = avg_xp_kl
                dict_to_log["UpdateStep"] = update_steps
                
                play_card_knowledge_names = ["Neither", "OnlyColor", "OnlyRank", "Both"]
                sp_play_card_knowledge = list(sp_play_card_knowledge)
                sp_play_card_knowledge[2] += sp_play_card_knowledge[3]
                sp_play_card_knowledge[1] += sp_play_card_knowledge[2]
                sp_play_card_knowledge[0] += sp_play_card_knowledge[1]
                for j in range(len(play_card_knowledge_names)):
                    for process_id in range(config["NUM_GPUS"]):
                        dict_to_log[play_card_knowledge_names[j] + "_" + str(process_id)] = sp_play_card_knowledge[j][process_id]
                print("Avg SP:", avg_sp_score)
                print("Avg XP:", avg_xp_score)
                print("Max SP:", max_sp_score)
                print("Min SP:", min_sp_score)
                now = time.time()
                print("Time for XP:", now-before_xp)

            # SAVE CHECKPOINT
            if config["TRAIN_POLICY"]:
                if config["CHECKPOINTS"] or i == config["NUM_OUTER_UPDATES"]-1:
                    print("Save ActorCritic Weights")
                    list_train_states = [jax.tree.map(lambda x: x[j], train_states) for j in range(config["NUM_GPUS"])]
                    for process_id, train_state in enumerate(list_train_states):
                        save_model_checkpoint(train_state, run_name, wandb_run_name, process_id)
            if config["TRAIN_BELIEF_MODEL"]:
                if config["CHECKPOINTS"] or i == config["NUM_OUTER_UPDATES"]-1:
                    print("Save BeliefModel Weights")
                    list_bm_train_states = [jax.tree.map(lambda x: x[j], bm_train_states) for j in range(config["NUM_GPUS"])]
                    for process_id, bm_train_state in enumerate(list_bm_train_states):
                        save_model_checkpoint(bm_train_state, run_name, wandb_run_name, process_id)
            
            # COMPUTE REMAINING TIME
            elapsed = now - last_call_time
            if i >= 2:
                remaining = elapsed * (config["NUM_OUTER_UPDATES"] - i+1) / 3600
            else:
                remaining = 0
            dict_to_log["Time remaining in hours"] = remaining
            last_call_time = time.time()

            # WANDB
            wandb.log(dict_to_log)

        return {"update_states": update_states}
    
    return train

@hydra.main(version_base=None, config_path="config", config_name="obl_hanabi")
def main(config):
    config = OmegaConf.to_container(config)
    if config["SEED"] != False:
        seed = config["SEED"]        
    else:
        seed = np.random.randint(10000)
    rng = jax.random.PRNGKey(seed)

    # Set run name and tags
    def _get_run_name_and_tags(config):
        run_name= ""
        tags = []
        num_agents = str(config["ENV_KWARGS"]["num_agents"]) + "P"
        if config["TRAIN_POLICY"]:
            if config["OBL"]:
                run_name += "OBL_" + num_agents
                tags += ["OBL", num_agents]
                if config["BELIEF_MODEL_FOLDER"] == False:
                    run_name += "_L1"
                    tags += ["L1", "CountBM"]
                else:
                    belief_model_filename = sorted([f for f in os.listdir(config["BELIEF_MODEL_FOLDER"]) if f.endswith(".safetensors")])[0]
                    level = str(int(re.search(r'L\d+', belief_model_filename).group()[1]) + 1)
                    run_name += "_L" + level
                    tags += ["L" + level]
                if config["CRITIC_CENTRALIZED"]:
                    run_name += "_CENTRAL"
                    tags += ["CENTRAL"]
                else:
                    run_name += "_LOCAL"
                    tags += ["LOCAL"]
            elif config["CRITIC_CENTRALIZED"]:
                run_name+= "MAPPO"
                tags += ["MAPPO"]
                run_name+= "_" + num_agents
                tags += [num_agents]
            else:
                run_name+= "IPPO"
                run_name+= "_" + num_agents
                tags += [num_agents]
                tags += ["IPPO"]

            if config["RNN"] not in ["LSTM", "GRU"]:
                run_name+= "_FF"
                tags += "FF"
            else:
                run_name += "_" + config["RNN"]
                tags += [config["RNN"]]

        elif config["TRAIN_BELIEF_MODEL"]:
            run_name += "BM_" + num_agents
            tags += ["BM", num_agents]
            if config["ACTOR_CRITIC_FOLDER"] == False:
                run_name += "_L0"
                tags += ["L0"]
            else:
                actor_critic_filename = sorted([f for f in os.listdir(config["BELIEF_MODEL_FOLDER"]) if f.endswith(".safetensors")])[0]
                level = str(int(re.search(r'L\d+', actor_critic_filename).group()[1]) + 1)
                run_name += "_L" + level
                tags += ["L" + level]

        return run_name, tags
    
    run_name, tags = _get_run_name_and_tags(config)
    wandb_run_name = run_name + config["SUFFIX"]

    run = wandb.init(
        project=config["PROJECT"],
        name=wandb_run_name,
        tags=tags,
        config=config,
        mode=config["WANDB_MODE"],
    )
    wandb.run.log_code(".")
    with jax.disable_jit(False):
        train_jit = make_train(config, run_name, wandb_run_name)
        out = train_jit(rng)
        run.finish()

if __name__ == "__main__":
    main()
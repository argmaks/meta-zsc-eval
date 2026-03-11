import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal, xavier_uniform, xavier_normal
from typing import Sequence, Dict
import distrax
import functools
import jax



class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        
        # Select kernel initializer based on config
        kernel_init_type = self.config.get("KERNEL_INIT", "orthogonal")
        if kernel_init_type == "xavier_uniform":
            init_sqrt2 = xavier_uniform()
            init_2 = xavier_uniform()
            init_001 = xavier_uniform()
            init_1 = xavier_uniform()
        elif kernel_init_type == "xavier_normal":
            init_sqrt2 = xavier_normal()
            init_2 = xavier_normal()
            init_001 = xavier_normal()
            init_1 = xavier_normal()
        else:  # default to orthogonal
            init_sqrt2 = orthogonal(np.sqrt(2))
            init_2 = orthogonal(2)
            init_001 = orthogonal(0.01)
            init_1 = orthogonal(1.0)
        
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=init_sqrt2, bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=init_2, bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=init_001, bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=init_2, bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=init_1, bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)



def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}





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
        # obs_TBO, hand_TBY, dones_TB, avail_actions_TBA = x_TB_
        obs_TBO, dones_TB, avail_actions_TBA = x_TB_

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

        pi = distrax.Categorical(logits=action_logits_TBA)

        return hidden_state_LBW_, pi, critic_TB
        
    @staticmethod
    def initialize_hidden_state(rnn, num_rnn_layers, batch_size, hidden_size):
        if rnn == "LSTM":
            initial_hidden_state_LBW_ = ScannedMultiLayerLSTM.initialize_hidden_state(num_rnn_layers, batch_size, hidden_size)
        elif rnn == "GRU":
            initial_hidden_state_LBW_ = ScannedMultiLayerGRU.initialize_hidden_state(num_rnn_layers, batch_size, hidden_size)
        else:
            initial_hidden_state_LBW_ = ScannedMultiLayerGRU.initialize_hidden_state(1, batch_size, 1)
        return initial_hidden_state_LBW_
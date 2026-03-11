import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
import flax.serialization as fs
import distrax

def identity_init():
    def init(key, shape, dtype=jnp.float32):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Shape must be square for a permutation matrix, and act on 21 dimensional action space.")
        return jnp.eye(shape[0], dtype=dtype)
    return init

class DenseLayerConfig(NamedTuple):
    units: int
    kernel: Any

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    def setup(self):
        self.layer1 = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.layer2 = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.layer3 = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.layer4 = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.layer5 = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x):
        obs, dones, avail_actions, out_permutations = x

        embedding = self.layer1(obs)
        embedding = nn.relu(embedding)

        actor_mean = self.layer2(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = self.layer3(actor_mean)

        transformed_actor_mean = jax.vmap(lambda a, p: jnp.dot(a, p))(actor_mean.reshape(-1,actor_mean.shape[-1]), out_permutations).reshape(actor_mean.shape)

        unavail_actions = 1 - avail_actions
        action_logits = transformed_actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = self.layer4(embedding)
        critic = nn.relu(critic)
        critic = self.layer5(critic)

        return pi, jnp.squeeze(critic, axis=-1)

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}
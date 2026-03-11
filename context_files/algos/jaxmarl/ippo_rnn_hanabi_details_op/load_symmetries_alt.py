import os
import pickle
from typing import List, Tuple

import numpy as np
import jax.numpy as jnp

def load_permutations() -> Tuple[np.ndarray, np.ndarray]:
    """Load observation and action permutation indices corresponding to the DecPOMDP symmetries. 

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two jax.numpy arrays with shapes
        ``(num_permutations, obs_dim)``  and
        ``(num_permutations, action_dim)`` corresponding to
        observation and action permutation indices, respectively.
        The indices can be use to map the unpermuted observation/action to the permuted observation/action by indexing the unpermuted observation/action with the indices.
    """


    directory = os.path.join(os.path.dirname(__file__), 'hanabi_symmetries')
    with open(os.path.join(directory, 'obs_permutation_indices.pkl'), 'rb') as f:
        obs_permutation_indices = pickle.load(f)
        obs_permutation_indices = jnp.array(obs_permutation_indices, dtype=jnp.int32)
    with open(os.path.join(directory, 'action_permutation_indices.pkl'), 'rb') as f:
        action_permutation_indices = pickle.load(f)   
        action_permutation_indices = jnp.array(action_permutation_indices, dtype=jnp.int32)

    return obs_permutation_indices, action_permutation_indices



if __name__ == "__main__":
    o, a = load_permutations()
    print(o.shape)
    print(a.shape)

    





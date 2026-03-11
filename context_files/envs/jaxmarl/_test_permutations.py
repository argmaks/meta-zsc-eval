# %%
import jax.numpy as jnp
import os
import pickle

# %%
# Equivalence of matrix vs indices based permutation calculation
matrix_permutation = jnp.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
indices_permutation = jnp.array([1,2,3,0])
inverse_indices_permutation = jnp.argsort(indices_permutation)
dummy_vector = jnp.array([1,2,3,4])

matrix_permuted = jnp.dot(matrix_permutation, dummy_vector)
indices_permuted = jnp.take_along_axis(dummy_vector, indices_permutation, axis=-1)

print("permute from the left (Px) = take along axis")
jnp.all(matrix_permuted == indices_permuted)
# %%
print("construct matrix from indices via P[i, indices[i]] = 1")
print(jnp.all(jnp.array([matrix_permutation[i, indices_permutation[i]] for i in range(matrix_permutation.shape[0])]) == jnp.ones(matrix_permutation.shape[0])))

# %%
print("hence a permutation matrix constructed from indices via P[i, indices[i]] = 1 is a left permutation matrix Px")

# %%
matrix_permuted = jnp.dot(dummy_vector, matrix_permutation)
indices_permuted = jnp.take_along_axis(dummy_vector, inverse_indices_permutation, axis=-1)

print("permute from the right (xP) = take along axis of the inverse")
jnp.all(matrix_permuted == indices_permuted)

# %%
directory = os.path.join(os.path.dirname(__file__), 'hanabi_symmetries')
with open(os.path.join(directory, 'obs_permutation_indices.pkl'), 'rb') as f:
    # LEFT PERMUTATION INDICES
    obs_permutation_indices = pickle.load(f)
    obs_permutation_indices = jnp.array(obs_permutation_indices, dtype=jnp.int32)
with open(os.path.join(directory, 'action_permutation_indices.pkl'), 'rb') as f:
    # LEFT PERMUTATION INDICES
    action_permutation_indices = pickle.load(f)   
    action_permutation_indices = jnp.array(action_permutation_indices, dtype=jnp.int32)
with open(os.path.join(directory, 'obs_permutation_matrices.pkl'), 'rb') as f:
    # LEFT PERMUTATION MATRICES
    obs_permutation_matrices = pickle.load(f)
    obs_permutation_matrices = jnp.array(obs_permutation_matrices, dtype=jnp.int32)
with open(os.path.join(directory, 'action_permutation_matrices.pkl'), 'rb') as f:
    # RIGHT PERMUTATION MATRICES OR INVERSE PERMUTATION MATRICES
    action_permutation_matrices = pickle.load(f)
    action_permutation_matrices = jnp.array(action_permutation_matrices, dtype=jnp.int32)

inverse_action_permutation_indices = jnp.argsort(action_permutation_indices, axis=-1)
# %%


obs = jnp.arange(658)
action = jnp.arange(21)

perm_idx = 68
# %%
# GFPPOY CALCULATION
"""
def transform_obs(obs, in_permutation):
        transformed_obs = jnp.dot(obs, in_permutation)
        return transformed_obs
obs_batch = jax.vmap(transform_obs, in_axes=(0, 0))(
                obs_batch.reshape(-1, obs_batch.shape[-1]),
                in_permutations[shuffle_colour_indices]
            ).reshape(obs_batch.shape)
"""
obs_permuted = jnp.dot(obs, obs_permutation_matrices[perm_idx])
# obs_permuted
# %%
"""
jax.vmap(lambda a, p: jnp.dot(a, p))(actor_mean.reshape(-1,actor_mean.shape[-1]), out_permutations).reshape(actor_mean.shape)
"""
action_permuted = jnp.dot(action, action_permutation_matrices[perm_idx])
# action_permuted
# %%
# EQUIVALENTCALCULATION JUST USING INDICES
obs_permuted_indices = jnp.take_along_axis(obs, jnp.argsort(obs_permutation_indices[perm_idx]), axis=-1)
# obs_permuted_indices
# %%
action_permuted_indices = jnp.take_along_axis(action, action_permutation_indices[perm_idx], axis=-1) 
# action_permuted_indices
# %%
jnp.all(obs_permuted_indices == obs_permuted)

# %%
jnp.all(action_permuted_indices == action_permuted)
# %%
# CALCULATION IN MY CURRENT CODE
"""
obs_batch = jnp.take_along_axis(obs_batch, obs_permutations[shuffle_colour_indices], axis=-1)
"""
obs_permuted_my_code = jnp.take_along_axis(obs, obs_permutation_indices[perm_idx], axis=-1)
obs_permuted_my_code
# %%
"""
unpermuted_action = jnp.take_along_axis(inverse_action_permutations[shuffle_colour_indices], action.reshape(action.shape[-1], -1),axis=-1).reshape(action.shape)
"""
action_permuted_my_code = jnp.array([jnp.take_along_axis(inverse_action_permutation_indices[perm_idx], action,axis=-1).reshape(action.shape) for action in jnp.arange(21).reshape(1, -1)]).squeeze()
# action_permuted_my_code
# %%
jnp.all(obs_permuted_my_code == obs_permuted)
# %%
jnp.all(action_permuted_my_code == action_permuted)

# %%
# CALCULATION IN MY CURRENT CODE BUT INVERTED
obs_permuted_my_code_inverted = jnp.take_along_axis(obs, jnp.argsort(obs_permutation_indices[perm_idx]), axis=-1)
# obs_permuted_my_code_inverted
# %%
action_permuted_my_code_inverted = jnp.array([jnp.take_along_axis(jnp.argsort(inverse_action_permutation_indices[perm_idx]), action,axis=-1).reshape(action.shape) for action in jnp.arange(21).reshape(1, -1)]).squeeze()
# action_permuted_my_code_inverted
# %%
jnp.all(obs_permuted_my_code_inverted == obs_permuted)
# %%
jnp.all(action_permuted_my_code_inverted == action_permuted)
# %%

print(f"Number of permutations: {len(obs_permutation_indices)}")

# Test all permutations
all_results = []
for perm_idx in range(len(obs_permutation_indices)):
    results = {}
    results['perm_idx'] = perm_idx
    
    # GFPPOY CALCULATION
    obs_permuted = jnp.dot(obs, obs_permutation_matrices[perm_idx])
    action_permuted = jnp.dot(action, action_permutation_matrices[perm_idx])
    
    # EQUIVALENT CALCULATION JUST USING INDICES
    obs_permuted_indices = jnp.take_along_axis(obs, jnp.argsort(obs_permutation_indices[perm_idx]), axis=-1)
    action_permuted_indices = jnp.take_along_axis(action, action_permutation_indices[perm_idx], axis=-1)
    
    results['obs_indices_eq_matrix'] = bool(jnp.all(obs_permuted_indices == obs_permuted))
    results['action_indices_eq_matrix'] = bool(jnp.all(action_permuted_indices == action_permuted))
    
    # CALCULATION IN MY CURRENT CODE
    obs_permuted_my_code = jnp.take_along_axis(obs, obs_permutation_indices[perm_idx], axis=-1)
    action_permuted_my_code = jnp.array([jnp.take_along_axis(inverse_action_permutation_indices[perm_idx], action,axis=-1).reshape(action.shape) for action in jnp.arange(21).reshape(1, -1)]).squeeze()
    
    results['obs_my_code_eq_matrix'] = bool(jnp.all(obs_permuted_my_code == obs_permuted))
    results['action_my_code_eq_matrix'] = bool(jnp.all(action_permuted_my_code == action_permuted))
    
    # CALCULATION IN MY CURRENT CODE BUT INVERTED
    obs_permuted_my_code_inverted = jnp.take_along_axis(obs, jnp.argsort(obs_permutation_indices[perm_idx]), axis=-1)
    action_permuted_my_code_inverted = jnp.array([jnp.take_along_axis(jnp.argsort(inverse_action_permutation_indices[perm_idx]), action,axis=-1).reshape(action.shape) for action in jnp.arange(21).reshape(1, -1)]).squeeze()
    
    results['obs_my_code_inverted_eq_matrix'] = bool(jnp.all(obs_permuted_my_code_inverted == obs_permuted))
    results['action_my_code_inverted_eq_matrix'] = bool(jnp.all(action_permuted_my_code_inverted == action_permuted))
    
    all_results.append(results)

# Print summary
print("\n=== Summary of All Permutations ===")
for key in ['obs_indices_eq_matrix', 'action_indices_eq_matrix', 
            'obs_my_code_eq_matrix', 'action_my_code_eq_matrix',
            'obs_my_code_inverted_eq_matrix', 'action_my_code_inverted_eq_matrix']:
    all_pass = all(r[key] for r in all_results)
    num_pass = sum(r[key] for r in all_results)
    print(f"{key}: {num_pass}/{len(all_results)} pass - {'✓ ALL PASS' if all_pass else '✗ SOME FAIL'}")
    if not all_pass:
        failing = [r['perm_idx'] for r in all_results if not r[key]]
        print(f"  Failing indices: {failing[:10]}{'...' if len(failing) > 10 else ''}")

print(f"\nTotal permutations tested: {len(all_results)}")


# %% 
# IN GFPPOY CODE
# TRANSITION is:
"""
transition = Transition(
                    done_batch, 
                    action.squeeze(), # action in unpermuted space
                    value.squeeze(), 
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(), # log_prob in unpermuted space
                    obs_batch, # obs in permuted space
                    info,
                    avail_actions,
                    shuffle_colour_indices
                )
"""
# then for loss calculation the agent learns from:
# - permuted obs
# - values for the permuted obs
# - logprobs in unpermuted space
# - action in unpermuted space
# - reward in unpermuted space

# IN MY CODE
"""
transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(), # action in permuted space
                    value.squeeze(), # values for the permuted obs
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(), # logprobs in permuted space
                    obs_batch, # obs in permuted space
                    info,
                    avail_actions
                )
"""

# then for loss calculation the agent learns from:
# - permuted obs
# - values for the permuted obs
# - logprobs in permuted space
# - action in permuted space
# - reward in unpermuted space

# COMPARISON
# so they are looking like they are different but both learn to output actions from obs in the end and since the permutations are applied inside the network in training the signal gets backpropagated to the permuted space so all in all the learning should be the same though the output space differs.
# edited from https://github.com/gfppoy/expected-return-symmetries/real_op_symmetries.py

"""
Get Dec-POMDP symmetry transformations for 2-player Hanabi.
"""
# %%
import jax
import jax.numpy as jnp
import numpy as np
import itertools
import pickle
import jaxmarl
import yaml
import os
# %%
# load the config
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = config['runner']['algo_config']
config["ENV_KWARGS"]["num_cards_of_rank"] = np.array(config["ENV_KWARGS"]["num_cards_of_rank"])
print(config["ENV_KWARGS"])
num_agents = config["ENV_KWARGS"]["num_agents"]
assert num_agents == 2, "The calculation below assumes 2 agents"
num_colors = config["ENV_KWARGS"]["num_colors"]
num_ranks = config["ENV_KWARGS"]["num_ranks"]
hand_size = config["ENV_KWARGS"]["hand_size"]
max_info_tokens = config["ENV_KWARGS"]["max_info_tokens"]
max_life_tokens = config["ENV_KWARGS"]["max_life_tokens"]
num_cards_of_rank = config["ENV_KWARGS"]["num_cards_of_rank"]
deck_size = num_colors * num_cards_of_rank.sum()
num_moves = 2 * hand_size + (num_agents - 1) * (num_colors + num_ranks) + 1
# make the environment
env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

# %%
obs_permutations_matrices = []
action_permutations_matrices = []

# %%
perms = list(itertools.permutations(list(range(num_colors)))) # all possible colour permutations

# NOTE: though the config above, the calculation below assumes 2 agents
# %%
for perm_num, colour_map in enumerate(perms):
    indices = np.zeros(env.obs_size) # this is the permutation map

    """
    self.hands_n_feats = (
            (self.num_agents - 1) * self.hand_size * self.num_colors * self.num_ranks
            + self.num_agents  # hands of all the other agents + agents' missing cards
        )
    """

    for card in range(hand_size): # hand card index
        for colour in range(num_colors): # card colour
            for rank in range(num_ranks): # card rank
                indices[card * num_colors * num_ranks + colour * num_ranks + rank] = card * num_colors * num_ranks + colour_map[colour] * num_ranks + rank # permute the indices according to the colour map
                
    indices[hand_size * num_colors * num_ranks] = hand_size * num_colors * num_ranks # keep as is
    indices[hand_size * num_colors * num_ranks + 1] = hand_size * num_colors * num_ranks + 1 # keep as is

    curr_idx = hand_size * num_colors * num_ranks + 2 # move on to the board features
    

    """
    self.board_n_feats = (
            (self.deck_size - self.num_agents * self.hand_size) # deck-initial cards, thermometer
            + self.num_colors * self.num_ranks  # fireworks, OH
            + self.max_info_tokens  # info tokens, OH
            + self.max_life_tokens    # life tokens, OH
        )
    """

    # keep the deck features as is
    for deck_feat in range(deck_size - 2 * hand_size):
        indices[curr_idx] = curr_idx
        curr_idx += 1
        
    for firework in range(num_colors): # firework index
        for rank in range(num_ranks): # firework rank
            indices[curr_idx + firework * num_ranks + rank] = curr_idx + colour_map[firework] * num_ranks + rank # permute the indices according to the colour map
    curr_idx += num_colors * num_ranks # move on to the info tokens
            
    # keep the info tokens as is
    for token in range(max_info_tokens + max_life_tokens): # token index
        indices[curr_idx] = curr_idx
        curr_idx += 1

    """
    self.discards_n_feats = self.num_colors * self.num_cards_of_rank.sum()
    """

    # permute the discard features according to the colour map
    for discard_colour in range(num_colors): # discard colour index
        for rank in range(num_cards_of_rank.sum()): # discard rank
            indices[curr_idx + discard_colour * num_cards_of_rank.sum() + rank] = curr_idx + colour_map[discard_colour] * num_cards_of_rank.sum() + rank # permute the indices according to the colour map
    curr_idx += num_colors * num_cards_of_rank.sum() # move on to the last action features
    
    """
    self.last_action_n_feats = (
                self.num_agents # acting player index
                + 4  # move type
                + self.num_agents  # target player index
                + self.num_colors  # color revealed
                + self.num_ranks  # rank revealed
                + self.hand_size  # reveal outcome
                + self.hand_size   # position played/discared
                + self.num_colors * self.num_ranks # card played/discarded
                + 1   # card played score
                + 1   # card played added info toke
            )
    """


    # permute the last action features
    for more_last_action_feats in range(2+4+2): # leave acting player index, movetype, target player index as is
        indices[curr_idx] = curr_idx
        curr_idx += 1
        

    for colour_revealed in range(num_colors): # permute the colour revealed features according to the colour map
        indices[curr_idx + colour_revealed] = curr_idx + colour_map[colour_revealed] 
    curr_idx += num_colors 


    for more_last_action_feats in range(num_ranks + hand_size + hand_size): # leave rank revealed, reveal outcome, position played/discarded as is
        indices[curr_idx] = curr_idx
        curr_idx += 1
        
        
    for colour in range(num_colors): # permute the card played/discarded features according to the colour map
        for rank in range(num_ranks):
            indices[curr_idx + colour * num_ranks + rank] = curr_idx + colour_map[colour] * num_ranks + rank
    curr_idx += num_colors * num_ranks
        
    for more_last_action_feats in range(1+1): # leave card player scored and card played added info tokens as is
        indices[curr_idx] = curr_idx
        curr_idx += 1
        

    """
    self.v0_belief_n_feats = (
            self.num_agents
            * self.hand_size
            * (  # feats for each card, mine and other players
                self.num_colors * self.num_ranks + self.num_colors + self.num_ranks
            )  # 35 feats per card (25deductions+10hints)
        )
    """

    # permute the beliefs.
    for card in range(2 * hand_size):
        # possible card
        for colour in range(num_colors):
            for rank in range(num_ranks):
                indices[curr_idx + colour * num_ranks + rank] = curr_idx + colour_map[colour] * num_ranks + rank
        curr_idx += num_colors * num_ranks
        # colour hinted
        for colour in range(num_colors):
            indices[curr_idx + colour] = curr_idx + colour_map[colour]
        curr_idx += num_colors
        # rank hinted
        for rank in range(num_ranks):
            indices[curr_idx] = curr_idx
            curr_idx += 1

    # create the permutation map for observations (unpermuted to permuted)
    obs_permutation_matrix = jnp.zeros((env.obs_size, env.obs_size), dtype=jnp.float32)
    for i in range(env.obs_size):
        obs_permutation_matrix = obs_permutation_matrix.at[i, int(indices[i])].set(1.)

    obs_permutations_matrices.append(obs_permutation_matrix)

    # permute actions
    indices = np.zeros(num_moves)
    curr_idx = 0
    for i in range(2 * hand_size): # keep discard/play indices as is
        indices[i] = curr_idx
        curr_idx += 1
    for i in range(num_colors): # permute the colour of hints
        indices[curr_idx + i] = curr_idx + colour_map[i]
    curr_idx += num_colors
    for i in range(num_ranks + 1): # keep ranks and noop as is
        indices[curr_idx] = curr_idx
        curr_idx += 1

    # create the permutation map for actions(unpermuted to permuted)
    action_permutation_matrix = jnp.zeros((num_moves, num_moves), dtype=jnp.float32)
    for i in range(num_moves):
        action_permutation_matrix = action_permutation_matrix.at[int(indices[i]), i].set(1.)

    action_permutations_matrices.append(action_permutation_matrix)

    

# %%
obs_permutations_matrices = np.array(obs_permutations_matrices)
action_permutations_matrices = np.array(action_permutations_matrices)

# %%
obs_permutations_matrices.shape
# %%
action_permutations_matrices.shape
# %%
import os
hanabi_symmetries_dir = os.path.join(os.path.dirname(__file__), 'hanabi_symmetries')
os.makedirs(hanabi_symmetries_dir, exist_ok=True)

with open(f'{hanabi_symmetries_dir}/obs_permutation_matrices.pkl', 'wb') as f:
    pickle.dump(obs_permutations_matrices, f)

with open(f'{hanabi_symmetries_dir}/action_permutation_matrices.pkl', 'wb') as f:
    pickle.dump(action_permutations_matrices, f)
# %%
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import chex
from functools import partial

def get_card_color_and_rank(card: chex.Array) -> Tuple[int, int]:
    """
    Extract color and rank from a one-hot encoded card.
    
    Args:
        card: One-hot encoded card of shape (num_colors, num_ranks)
        
    Returns:
        Tuple of (color_index, rank_index)
    """
    is_empty = ~card.any()
    color = jnp.argmax(card.sum(axis=1))
    rank = jnp.argmax(card.sum(axis=0))
    
    # Return -1 for empty cards, otherwise return the actual color/rank
    color = jnp.where(is_empty, -1, color)
    rank = jnp.where(is_empty, -1, rank)
    
    return color, rank

def is_card_playable(card: chex.Array, fireworks: chex.Array) -> bool:
    """
    Check if a card is playable right now.
    
    A card is playable if rank(card) = fireworks[color(card)] + 1
    """
    is_empty = ~card.any()
    color, rank = get_card_color_and_rank(card)
    
    # Get current firework level for this color (sum of ranks played)
    # Use jnp.where to handle the case where color might be -1 (empty card)
    current_firework_level = jnp.where(
        color >= 0,
        jnp.sum(fireworks[color]),
        0
    )
    
    # Card is playable if its rank is exactly the next needed rank and card is not empty
    is_playable = (rank == current_firework_level) & (~is_empty)
    return is_playable

def is_card_discardable(card: chex.Array, fireworks: chex.Array, 
                       deck: chex.Array, discard_pile: chex.Array,
                       num_cards_of_rank: jnp.ndarray) -> bool:
    """
    Check if a card is safely discardable.
    
    A card is discardable if:
    1. Its rank < current firework level for its color (already played/obsolete), OR
    2. Some prerequisite rank between current_firework_level and card's rank is impossible 
       to obtain (all copies are already played/discarded)
    """
    is_empty = ~card.any()
    color, rank = get_card_color_and_rank(card)
    
    # Get current firework level for this color (number of ranks completed)
    current_firework_level = jnp.where(
        color >= 0,
        jnp.sum(fireworks[color]),
        0
    )
    
    # Check condition 1: rank < current firework level (already obsolete)
    condition1 = (rank < current_firework_level) & (~is_empty)
    
    # Check condition 2: prerequisite ranks are impossible to obtain
    # We only need to check ranks between current_firework_level and rank (exclusive)
    # If rank == current_firework_level, the card is playable, not discardable
    def check_prereq_impossible(prereq_rank):
        # Count played cards of this specific rank for this color (0 or 1)
        # fireworks[color, rank] is 1 if that rank has been played
        played_count = jnp.where(
            color >= 0,
            fireworks[color, prereq_rank],
            0
        )
        
        # Count discarded cards of this rank for this color
        discarded_count = jnp.where(
            color >= 0,
            jnp.sum(discard_pile[:, color, prereq_rank]),
            0
        )
        
        # Total cards of this rank that existed in the original deck
        total_cards_of_this_rank = num_cards_of_rank[prereq_rank]
        
        # Check if all copies are gone (played + discarded >= total)
        return played_count + discarded_count >= total_cards_of_this_rank
    
    # Check prerequisite ranks between current_firework_level and rank
    # For JAX compatibility, we'll check a fixed number of ranks (max 5 for standard Hanabi)
    max_rank = 5
    prereq_impossible = jnp.zeros(max_rank, dtype=bool)
    
    for i in range(max_rank):
        # Only check ranks that are:
        # 1. Greater than or equal to current_firework_level (not yet on fireworks)
        # 2. Less than the card's rank (prerequisite)
        # 3. Card is not empty and has valid color
        prereq_impossible = prereq_impossible.at[i].set(
            jnp.where(
                (i >= current_firework_level) & (i < rank) & (color >= 0) & (~is_empty),
                check_prereq_impossible(i),
                False
            )
        )
    
    condition2 = jnp.any(prereq_impossible) & (~is_empty)
    
    return condition1 | condition2

def classify_card_status(card: chex.Array, fireworks: chex.Array, 
                        deck: chex.Array, discard_pile: chex.Array,
                        num_cards_of_rank: jnp.ndarray) -> int:
    """
    Classify a single card and return status as integer (extended logic).
    
    Returns:
        0: playable
        1: discardable  
        2: unknown
    """
    is_empty = ~card.any()
    
    # Check if playable
    is_playable = is_card_playable(card, fireworks)
    
    # Check if discardable
    is_discardable = is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank)
    
    # Use jnp.where for conditional logic
    # Priority: playable > discardable > unknown
    status = jnp.where(
        is_playable,
        0,  # playable
        jnp.where(
            is_discardable,
            1,  # discardable
            2   # unknown
        )
    )
    
    # Empty cards are always unknown
    status = jnp.where(is_empty, 2, status)
    
    return status

def classify_card_status_original(card: chex.Array, fireworks: chex.Array) -> int:
    """
    Classify a single card using original logic (matching C++ implementation).
    
    Logic:
    - Playable: rank == firework level
    - Discardable: rank < firework level  
    - Unknown: rank > firework level
    
    Returns:
        0: playable
        1: discardable  
        2: unknown
    """
    is_empty = ~card.any()
    color, rank = get_card_color_and_rank(card)
    
    # Get current firework level for this color (number of ranks completed)
    current_firework_level = jnp.where(
        color >= 0,
        jnp.sum(fireworks[color]),
        0
    )

    
    # Original logic: simple comparison with firework level
    status = jnp.where(
        rank == current_firework_level,
        0,  # playable
        jnp.where(
            rank < current_firework_level,
            1,  # discardable
            2   # unknown
        )
    )
    
    # Empty cards are always unknown
    status = jnp.where(is_empty, 2, status)
    
    return status

@partial(jax.jit, static_argnums=(1,))
def extract_hand_status(env_states, discardable_logic: str = "extended"):
    """
    Extract hand status for all players across all environments.
    
    Args:
        env_states: Batch of environment states from JaxMARL Hanabi
        discardable_logic: Either "extended" or "original"
            - "extended": More sophisticated logic that checks if cards are truly discardable
                         (obsolete or impossible to play due to missing prerequisites)
            - "original": Simple logic matching C++ implementation (rank == firework: playable,
                         rank < firework: discardable, rank > firework: unknown)
        
    Returns:
        hand_status: Array of shape (num_envs * num_agents, hand_size * 3)
                    where each card gets a one-hot encoding of [playable, discardable, unknown].
                    This is a flattened representation of 5 cards × 3 classes.
                    For loss computation, reshape to (..., 5, 3) to treat each card as 
                    a separate 3-class classification problem.
        hand_slot_mask: Array of shape (num_envs * num_agents, hand_size)
                    where 1.0 indicates a valid (non-empty) card slot, 0.0 indicates empty.
    """
    # Get environment parameters - assuming standard Hanabi setup
    num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])  # Standard Hanabi card distribution
    hand_size = 5  # Standard hand size
    num_agents = 2  # From config
    
    # Get batch dimensions
    batch_shape = env_states.player_hands.shape[0]  # num_envs
    
    # Select classification function based on discardable_logic
    use_original = (discardable_logic == "original")
    
    def process_single_env(env_idx):
        """Process a single environment and return hand status for both players."""
        env_state = jax.tree.map(lambda x: x[env_idx], env_states)
        
        # Extract game state components
        player_hands = env_state.player_hands  # (num_agents, hand_size, num_colors, num_ranks)
        fireworks = env_state.fireworks        # (num_colors, num_ranks)
        deck = env_state.deck                  # (deck_size, num_colors, num_ranks)
        discard_pile = env_state.discard_pile  # (deck_size, num_colors, num_ranks)
        
        def process_single_player(player_idx):
            """Process a single player's hand."""
            player_hand = player_hands[player_idx]  # (hand_size, num_colors, num_ranks)
            
            def process_single_card(card_idx):
                """Process a single card."""
                card = player_hand[card_idx]
                is_empty = ~card.any()
                
                # Choose classification method
                if use_original:
                    status = classify_card_status_original(card, fireworks)
                else:
                    status = classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
                
                # Convert to one-hot encoding: [playable, discardable, unknown]
                one_hot = jnp.zeros(3)
                one_hot = one_hot.at[status].set(1.0)
                
                # Mask: 1.0 if card is not empty, 0.0 if empty
                mask = jnp.where(is_empty, 0.0, 1.0)
                
                return one_hot, mask
            
            # Process all cards for this player using vmap
            card_statuses, card_masks = jax.vmap(process_single_card)(jnp.arange(hand_size))
            
            # Flatten statuses to (hand_size * 3,), keep masks as (hand_size,)
            return card_statuses.reshape(-1), card_masks
        
        # Process both players using vmap
        player_statuses, player_masks = jax.vmap(process_single_player)(jnp.arange(num_agents))
        
        return player_statuses, player_masks  # (num_agents, hand_size * 3), (num_agents, hand_size)
    
    # Process all environments using vmap
    all_env_statuses, all_env_masks = jax.vmap(process_single_env)(jnp.arange(batch_shape))
    
    # Reshape to (num_envs * num_agents, hand_size * 3) and (num_envs * num_agents, hand_size)
    hand_status = all_env_statuses.reshape(-1, hand_size * 3)
    hand_slot_mask = all_env_masks.reshape(-1, hand_size)
    
    return hand_status, hand_slot_mask

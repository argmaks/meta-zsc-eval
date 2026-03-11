import jax
import jax.numpy as jnp
import numpy as np
import sys
import traceback
from pathlib import Path

# # Add JaxMARL to path
# jaxmarl_path = Path(__file__).parent.parent.parent.parent.parent / "JaxMARL"
# sys.path.insert(0, str(jaxmarl_path))

from jaxmarl.environments.hanabi import Hanabi
import aux


class TestCardColorAndRank:
    """Test get_card_color_and_rank function."""
    
    def test_valid_card(self):
        """Test extracting color and rank from a valid card."""
        # Create a card: Red 2 (color=0, rank=1)
        card = jnp.zeros((5, 5))
        card = card.at[0, 1].set(1.0)
        
        color, rank = aux.get_card_color_and_rank(card)
        assert color == 0
        assert rank == 1
    
    def test_empty_card(self):
        """Test that empty cards return -1."""
        card = jnp.zeros((5, 5))
        
        color, rank = aux.get_card_color_and_rank(card)
        assert color == -1
        assert rank == -1
    
    def test_different_colors_ranks(self):
        """Test various color and rank combinations."""
        for c in range(5):
            for r in range(5):
                card = jnp.zeros((5, 5))
                card = card.at[c, r].set(1.0)
                
                color, rank = aux.get_card_color_and_rank(card)
                assert color == c
                assert rank == r


class TestCardPlayable:
    """Test is_card_playable function."""
    
    def test_playable_card_rank_0(self):
        """Test that rank 0 cards are playable on empty fireworks."""
        # Red 1 (rank 0) should be playable on empty fireworks
        card = jnp.zeros((5, 5))
        card = card.at[0, 0].set(1.0)
        
        fireworks = jnp.zeros((5, 5))
        
        assert aux.is_card_playable(card, fireworks) == True
    
    def test_playable_card_rank_1(self):
        """Test that rank 1 cards are playable when rank 0 is played."""
        # Red 2 (rank 1) should be playable when Red 1 is on fireworks
        card = jnp.zeros((5, 5))
        card = card.at[0, 1].set(1.0)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)  # Red 1 played
        
        assert aux.is_card_playable(card, fireworks) == True
    
    def test_not_playable_wrong_rank(self):
        """Test that wrong rank cards are not playable."""
        # Red 3 (rank 2) is NOT playable when Red 1 is on fireworks
        card = jnp.zeros((5, 5))
        card = card.at[0, 2].set(1.0)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)  # Only Red 1 played
        
        assert aux.is_card_playable(card, fireworks) == False
    
    def test_empty_card_not_playable(self):
        """Test that empty cards are not playable."""
        card = jnp.zeros((5, 5))
        fireworks = jnp.zeros((5, 5))
        
        assert aux.is_card_playable(card, fireworks) == False
    
    def test_different_colors(self):
        """Test playability across different colors."""
        # Green 2 (color=2, rank=1) should be playable when Green 1 is played
        card = jnp.zeros((5, 5))
        card = card.at[2, 1].set(1.0)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[2, 0].set(1.0)  # Green 1 played
        fireworks = fireworks.at[0, 0].set(1.0)  # Red 1 played (shouldn't matter)
        
        assert aux.is_card_playable(card, fireworks) == True
    
    def test_playable_with_multiple_ranks(self):
        """Test playability when multiple ranks are played for a color."""
        # Yellow 4 (color=1, rank=3) should be playable when Yellow 1,2,3 are played
        card = jnp.zeros((5, 5))
        card = card.at[1, 3].set(1.0)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[1, 0].set(1.0)  # Yellow 1
        fireworks = fireworks.at[1, 1].set(1.0)  # Yellow 2
        fireworks = fireworks.at[1, 2].set(1.0)  # Yellow 3
        
        assert aux.is_card_playable(card, fireworks) == True
    
    def test_playable_last_rank(self):
        """Test playability of the last rank (5)."""
        # Blue 5 (color=4, rank=4) should be playable when Blue 1,2,3,4 are played
        card = jnp.zeros((5, 5))
        card = card.at[4, 4].set(1.0)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[4, 0].set(1.0)  # Blue 1
        fireworks = fireworks.at[4, 1].set(1.0)  # Blue 2
        fireworks = fireworks.at[4, 2].set(1.0)  # Blue 3
        fireworks = fireworks.at[4, 3].set(1.0)  # Blue 4
        
        assert aux.is_card_playable(card, fireworks) == True
    
    def test_not_playable_last_rank_incomplete(self):
        """Test that last rank is not playable if sequence incomplete."""
        # Blue 5 should NOT be playable when only Blue 1,2,3 are played
        card = jnp.zeros((5, 5))
        card = card.at[4, 4].set(1.0)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[4, 0].set(1.0)  # Blue 1
        fireworks = fireworks.at[4, 1].set(1.0)  # Blue 2
        fireworks = fireworks.at[4, 2].set(1.0)  # Blue 3
        
        assert aux.is_card_playable(card, fireworks) == False
    
    def test_playable_independent_colors(self):
        """Test that playability is independent across colors."""
        # White 2 (color=3, rank=1) should be playable even if other colors are ahead
        card = jnp.zeros((5, 5))
        card = card.at[3, 1].set(1.0)
        
        fireworks = jnp.zeros((5, 5))
        # Red is complete
        fireworks = fireworks.at[0, :].set(1.0)
        # White only has rank 0
        fireworks = fireworks.at[3, 0].set(1.0)
        
        assert aux.is_card_playable(card, fireworks) == True
    
    def test_not_playable_already_played(self):
        """Test that already played cards are not playable."""
        # Red 2 should NOT be playable when Red 1,2,3 are already played
        card = jnp.zeros((5, 5))
        card = card.at[0, 1].set(1.0)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)  # Red 1
        fireworks = fireworks.at[0, 1].set(1.0)  # Red 2
        fireworks = fireworks.at[0, 2].set(1.0)  # Red 3
        
        assert aux.is_card_playable(card, fireworks) == False


class TestCardDiscardable:
    """Test is_card_discardable function."""
    
    def test_discardable_already_played(self):
        """Test that cards with ranks lower than fireworks are discardable."""
        # Red 1 is discardable when Red 2 is already on fireworks
        card = jnp.zeros((5, 5))
        card = card.at[0, 0].set(1.0)  # Red 1
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)  # Red 1 played
        fireworks = fireworks.at[0, 1].set(1.0)  # Red 2 played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True
    
    def test_not_discardable_needed_card(self):
        """Test that needed cards are not discardable."""
        # Red 2 is NOT discardable when Red 1 is on fireworks and we need Red 2
        card = jnp.zeros((5, 5))
        card = card.at[0, 1].set(1.0)  # Red 2
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)  # Red 1 played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == False
    
    def test_discardable_impossible_prereq(self):
        """Test that cards become discardable when prerequisites are impossible."""
        # Red 3 is discardable if all Red 2s are discarded
        card = jnp.zeros((5, 5))
        card = card.at[0, 2].set(1.0)  # Red 3
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)  # Red 1 played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        # Discard all 2 copies of Red 2 (rank 1)
        discard_pile = discard_pile.at[0, 0, 1].set(1.0)
        discard_pile = discard_pile.at[1, 0, 1].set(1.0)
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True
    
    def test_empty_card_not_discardable(self):
        """Test that empty cards are not discardable."""
        card = jnp.zeros((5, 5))
        fireworks = jnp.zeros((5, 5))
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == False
    
    def test_not_discardable_partial_prereq_loss(self):
        """Test that cards are NOT discardable if only some prerequisite cards are discarded."""
        # Red 3 should NOT be discardable if only 1 of 2 Red 2s are discarded
        card = jnp.zeros((5, 5))
        card = card.at[0, 2].set(1.0)  # Red 3
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)  # Red 1 played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        discard_pile = discard_pile.at[0, 0, 1].set(1.0)  # Only 1 Red 2 discarded (out of 2)
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == False
    
    def test_discardable_last_rank_if_prereq_impossible(self):
        """Test that rank 5 cards become discardable if rank 4 is impossible."""
        # Yellow 5 (rank 4) is discardable if all Yellow 4s (rank 3) are discarded
        card = jnp.zeros((5, 5))
        card = card.at[1, 4].set(1.0)  # Yellow 5
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[1, 0].set(1.0)  # Yellow 1 played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        # Discard all Yellow 4s (rank 3, there are 2 of them)
        discard_pile = discard_pile.at[0, 1, 3].set(1.0)
        discard_pile = discard_pile.at[1, 1, 3].set(1.0)
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True
    
    def test_discardable_multiple_missing_prereqs(self):
        """Test discardable when multiple intermediate ranks are impossible."""
        # Green 5 is discardable if Green 2 is impossible (even if Green 3,4 exist)
        card = jnp.zeros((5, 5))
        card = card.at[2, 4].set(1.0)  # Green 5
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[2, 0].set(1.0)  # Green 1 played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        # Discard all Green 2s (rank 1)
        discard_pile = discard_pile.at[0, 2, 1].set(1.0)
        discard_pile = discard_pile.at[1, 2, 1].set(1.0)
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True
    
    def test_discardable_with_advanced_fireworks(self):
        """Test that obsolete cards are discardable even when other colors lag behind."""
        # Red 2 is discardable when Red 1,2,3,4,5 are all played, regardless of other colors
        card = jnp.zeros((5, 5))
        card = card.at[0, 1].set(1.0)  # Red 2
        
        fireworks = jnp.zeros((5, 5))
        # Complete Red firework
        fireworks = fireworks.at[0, :].set(1.0)
        # Other colors at various stages
        fireworks = fireworks.at[1, 0].set(1.0)  # Yellow 1
        # Blue, Green, White have nothing
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True
    
    def test_not_discardable_rank_1_start(self):
        """Test that rank 1 cards are NOT discardable at game start."""
        # Any rank 1 (rank 0) card should not be discardable at start
        card = jnp.zeros((5, 5))
        card = card.at[0, 0].set(1.0)  # Red 1
        
        fireworks = jnp.zeros((5, 5))
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == False
    
    def test_discardable_rank_1_after_played(self):
        """Test that rank 1 cards become discardable after one is played."""
        # Red 1 becomes discardable once a Red 1 is on the fireworks
        card = jnp.zeros((5, 5))
        card = card.at[0, 0].set(1.0)  # Red 1
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)  # Red 1 played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True
    
    def test_discardable_immediate_prereq_vs_distant_prereq(self):
        """Test that a card is discardable if immediate prerequisite (rank-1) is impossible."""
        # White 3 is discardable if all White 2s are gone, even if White 1 exists
        card = jnp.zeros((5, 5))
        card = card.at[3, 2].set(1.0)  # White 3 (rank 2)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[3, 0].set(1.0)  # White 1 played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        # All White 2s (rank 1) are discarded
        discard_pile = discard_pile.at[0, 3, 1].set(1.0)
        discard_pile = discard_pile.at[1, 3, 1].set(1.0)
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True
    
    def test_discardable_combination_played_and_discarded(self):
        """Test when prerequisite cards are both played and discarded (all gone)."""
        # Blue 4 should be discardable if all Blue 3s are played+discarded
        card = jnp.zeros((5, 5))
        card = card.at[4, 3].set(1.0)  # Blue 4 (rank 3)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[4, 0].set(1.0)  # Blue 1 played
        fireworks = fireworks.at[4, 1].set(1.0)  # Blue 2 played
        fireworks = fireworks.at[4, 2].set(1.0)  # Blue 3 played (one copy on fireworks)
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        # One Blue 3 discarded (1 played + 1 discarded = 2 total, all gone)
        discard_pile = discard_pile.at[0, 4, 2].set(1.0)
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # Blue 4 is still playable since Blue 3 is on fireworks
        # It should NOT be discardable
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == False
    
    def test_discardable_rank_5_card(self):
        """Test special case of rank 5 (only 1 copy exists)."""
        # Red 5 is discardable if Red 4 is impossible
        card = jnp.zeros((5, 5))
        card = card.at[0, 4].set(1.0)  # Red 5 (rank 4)
        
        fireworks = jnp.zeros((5, 5))
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        # Discard all Red 4s (rank 3)
        discard_pile = discard_pile.at[0, 0, 3].set(1.0)
        discard_pile = discard_pile.at[1, 0, 3].set(1.0)
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True


class TestClassifyCardStatus:
    """Test classify_card_status function."""
    
    def test_classify_playable(self):
        """Test classification of playable cards."""
        # Red 1 on empty fireworks
        card = jnp.zeros((5, 5))
        card = card.at[0, 0].set(1.0)
        
        fireworks = jnp.zeros((5, 5))
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 0  # playable
    
    def test_classify_discardable(self):
        """Test classification of discardable cards."""
        # Red 1 when Red 1 and Red 2 are already played
        card = jnp.zeros((5, 5))
        card = card.at[0, 0].set(1.0)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)
        fireworks = fireworks.at[0, 1].set(1.0)
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 1  # discardable
    
    def test_classify_unknown(self):
        """Test classification of unknown cards."""
        # Red 3 when only Red 1 is played (Red 2 is still possible)
        card = jnp.zeros((5, 5))
        card = card.at[0, 2].set(1.0)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 2  # unknown
    
    def test_classify_empty(self):
        """Test that empty cards are classified as unknown."""
        card = jnp.zeros((5, 5))
        fireworks = jnp.zeros((5, 5))
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 2  # unknown
    
    def test_playable_takes_priority(self):
        """Test that playable status takes priority over discardable."""
        # This shouldn't happen in practice, but test the priority logic
        card = jnp.zeros((5, 5))
        card = card.at[0, 0].set(1.0)  # Red 1
        
        fireworks = jnp.zeros((5, 5))  # Empty fireworks, so Red 1 is playable
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 0  # playable
    
    def test_classify_with_multiple_colors(self):
        """Test classification with complex fireworks state."""
        # White 3 when White 1,2 are played - should be playable
        card = jnp.zeros((5, 5))
        card = card.at[3, 2].set(1.0)  # White 3
        
        fireworks = jnp.zeros((5, 5))
        # Red complete
        fireworks = fireworks.at[0, :].set(1.0)
        # White at rank 2
        fireworks = fireworks.at[3, 0].set(1.0)
        fireworks = fireworks.at[3, 1].set(1.0)
        # Yellow at rank 0
        fireworks = fireworks.at[1, 0].set(1.0)
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 0  # playable
    
    def test_classify_unknown_future_card(self):
        """Test that cards several ranks ahead are unknown."""
        # Green 5 when only Green 1,2 are played - should be unknown (not discardable yet)
        card = jnp.zeros((5, 5))
        card = card.at[2, 4].set(1.0)  # Green 5
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[2, 0].set(1.0)  # Green 1
        fireworks = fireworks.at[2, 1].set(1.0)  # Green 2
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 2  # unknown (Green 3,4 are still possible)
    
    def test_classify_last_playable_card(self):
        """Test classification of the last rank when it's playable."""
        # Blue 5 when Blue 1-4 are complete - should be playable
        card = jnp.zeros((5, 5))
        card = card.at[4, 4].set(1.0)  # Blue 5
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[4, 0].set(1.0)
        fireworks = fireworks.at[4, 1].set(1.0)
        fireworks = fireworks.at[4, 2].set(1.0)
        fireworks = fireworks.at[4, 3].set(1.0)
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 0  # playable
    
    def test_classify_discardable_due_to_broken_chain(self):
        """Test that cards become discardable when the prerequisite chain is broken."""
        # Yellow 4 when all Yellow 2s are discarded - should be discardable
        card = jnp.zeros((5, 5))
        card = card.at[1, 3].set(1.0)  # Yellow 4 (rank 3)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[1, 0].set(1.0)  # Yellow 1
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        # All Yellow 2s discarded (rank 1)
        discard_pile = discard_pile.at[0, 1, 1].set(1.0)
        discard_pile = discard_pile.at[1, 1, 1].set(1.0)
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 1  # discardable


class TestTrickyEdgeCases:
    """Test very tricky edge cases and corner scenarios."""
    
    def test_all_colors_different_progress(self):
        """Test with all 5 colors at different progress levels simultaneously."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :].set(1.0)  # Red complete (5/5)
        fireworks = fireworks.at[1, :4].set(1.0)  # Yellow at 4/5
        fireworks = fireworks.at[2, :2].set(1.0)  # Green at 2/5
        fireworks = fireworks.at[3, 0].set(1.0)   # White at 1/5
        # Blue at 0/5
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # Test each color
        # Red 1 - discardable (already complete)
        assert aux.is_card_discardable(
            jnp.zeros((5, 5)).at[0, 0].set(1.0), fireworks, deck, discard_pile, num_cards_of_rank
        ) == True
        
        # Yellow 5 - playable (Yellow at 4)
        assert aux.is_card_playable(
            jnp.zeros((5, 5)).at[1, 4].set(1.0), fireworks
        ) == True
        
        # Green 3 - playable (Green at 2)
        assert aux.is_card_playable(
            jnp.zeros((5, 5)).at[2, 2].set(1.0), fireworks
        ) == True
        
        # White 2 - playable (White at 1)
        assert aux.is_card_playable(
            jnp.zeros((5, 5)).at[3, 1].set(1.0), fireworks
        ) == True
        
        # Blue 1 - playable (Blue at 0)
        assert aux.is_card_playable(
            jnp.zeros((5, 5)).at[4, 0].set(1.0), fireworks
        ) == True
    
    def test_all_rank_1_cards_gone(self):
        """Test when all rank 1 cards of a color are gone - makes everything else discardable."""
        fireworks = jnp.zeros((5, 5))
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        
        # Discard all Red 1s (rank 0) - there are 3 of them
        discard_pile = discard_pile.at[0, 0, 0].set(1.0)
        discard_pile = discard_pile.at[1, 0, 0].set(1.0)
        discard_pile = discard_pile.at[2, 0, 0].set(1.0)
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # All Red cards (2,3,4,5) should now be discardable since Red 1 is impossible
        for rank in range(1, 5):
            card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
            assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True, \
                f"Red {rank+1} should be discardable when all Red 1s are gone"
    
    def test_last_copy_of_critical_card(self):
        """Test behavior with the last remaining copy of a critical card (rank 5)."""
        card = jnp.zeros((5, 5)).at[0, 4].set(1.0)  # Red 5 (only 1 copy exists)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :4].set(1.0)  # Red 1-4 played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # Red 5 should be playable (and is the ONLY copy)
        assert aux.is_card_playable(card, fireworks) == True
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == False
        
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 0  # playable
    
    def test_card_exactly_matches_firework_level(self):
        """Test edge case where card rank exactly equals firework level (already played)."""
        card = jnp.zeros((5, 5)).at[2, 2].set(1.0)  # Green 3 (rank 2)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[2, 0].set(1.0)  # Green 1
        fireworks = fireworks.at[2, 1].set(1.0)  # Green 2
        fireworks = fireworks.at[2, 2].set(1.0)  # Green 3 - same as our card!
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # Card is NOT playable (already played)
        assert aux.is_card_playable(card, fireworks) == False
        # Card IS discardable (obsolete)
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True
        # Status should be discardable
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 1
    
    def test_multiple_broken_chains_same_color(self):
        """Test when multiple ranks in the same chain are impossible."""
        card = jnp.zeros((5, 5)).at[1, 4].set(1.0)  # Yellow 5
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[1, 0].set(1.0)  # Yellow 1 played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        
        # Discard all Yellow 2s AND all Yellow 4s
        discard_pile = discard_pile.at[0, 1, 1].set(1.0)
        discard_pile = discard_pile.at[1, 1, 1].set(1.0)  # All Yellow 2s gone
        discard_pile = discard_pile.at[0, 1, 3].set(1.0)
        discard_pile = discard_pile.at[1, 1, 3].set(1.0)  # All Yellow 4s gone
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # Yellow 5 should be discardable (both 2 and 4 are impossible)
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True
    
    def test_second_to_last_rank_when_last_is_discarded(self):
        """Test rank 4 card when the rank 5 card is already discarded."""
        card = jnp.zeros((5, 5)).at[3, 3].set(1.0)  # White 4
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[3, :3].set(1.0)  # White 1,2,3 played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        discard_pile = discard_pile.at[0, 3, 4].set(1.0)  # White 5 discarded (only 1 copy)
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # White 4 should be playable (it can still be played, even if 5 is gone)
        assert aux.is_card_playable(card, fireworks) == True
        # White 4 should NOT be discardable (still useful to play)
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == False
        # Status should be playable
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 0
    
    def test_complete_color_all_cards_discardable(self):
        """Test that when a color is complete, all cards of that color are discardable."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[4, :].set(1.0)  # Blue complete
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # All Blue cards should be discardable
        for rank in range(5):
            card = jnp.zeros((5, 5)).at[4, rank].set(1.0)
            assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True
            assert aux.is_card_playable(card, fireworks) == False
            
            status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
            assert status == 1  # All should be classified as discardable
    
    def test_mixed_played_and_discarded_counting(self):
        """Test complex scenario with cards both played and discarded across ranks."""
        card = jnp.zeros((5, 5)).at[0, 3].set(1.0)  # Red 4 (rank 3)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)  # Red 1 played
        # Red 2 not played
        # Red 3 not played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        
        # For Red 2 (rank 1): play 1, discard 1 = 2/2 gone
        discard_pile = discard_pile.at[0, 0, 1].set(1.0)
        fireworks = fireworks.at[0, 1].set(1.0)  # One Red 2 played
        # But wait, this means Red 2 is on fireworks, so Red 3 can be played
        # Let's say Red 3 rank 2 instead
        
        # Actually, let me reconsider: if Red 1 is played, and all Red 2s are gone (played+discarded),
        # Then anything beyond Red 2 is impossible
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)  # Only Red 1 played
        
        # All Red 2s are discarded (not played)
        discard_pile = discard_pile.at[0, 0, 1].set(1.0)
        discard_pile = discard_pile.at[1, 0, 1].set(1.0)
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # Red 4 should be discardable since Red 2 is impossible
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == True
    
    def test_zero_fireworks_all_rank_1_playable(self):
        """Test that with empty fireworks, all rank 1 cards are playable across all colors."""
        fireworks = jnp.zeros((5, 5))
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # All rank 0 (value 1) cards should be playable
        for color in range(5):
            card = jnp.zeros((5, 5)).at[color, 0].set(1.0)
            assert aux.is_card_playable(card, fireworks) == True
            assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == False
            
            status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
            assert status == 0  # playable
    
    def test_batch_classification_all_ranks_same_color(self):
        """Test classifying all 5 ranks of the same color simultaneously."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[2, 0].set(1.0)  # Green 1 played
        fireworks = fireworks.at[2, 1].set(1.0)  # Green 2 played
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # Create all 5 Green cards
        cards = jnp.stack([jnp.zeros((5, 5)).at[2, r].set(1.0) for r in range(5)])
        
        # Classify all at once using vmap
        statuses = jax.vmap(
            lambda c: aux.classify_card_status(c, fireworks, deck, discard_pile, num_cards_of_rank)
        )(cards)
        
        # Green 1: discardable (rank < firework level)
        assert statuses[0] == 1
        # Green 2: discardable (rank < firework level)
        assert statuses[1] == 1
        # Green 3: playable (rank == firework level)
        assert statuses[2] == 0
        # Green 4: unknown (rank > firework level, but 3 is still possible)
        assert statuses[3] == 2
        # Green 5: unknown (rank > firework level)
        assert statuses[4] == 2
    
    def test_near_end_game_scenario(self):
        """Test classification near end game with most cards played."""
        fireworks = jnp.zeros((5, 5))
        # 4 colors complete, 1 color at rank 3
        fireworks = fireworks.at[0, :].set(1.0)    # Red complete
        fireworks = fireworks.at[1, :].set(1.0)    # Yellow complete
        fireworks = fireworks.at[2, :].set(1.0)    # Green complete
        fireworks = fireworks.at[3, :].set(1.0)    # White complete
        fireworks = fireworks.at[4, :3].set(1.0)   # Blue at 3
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # Blue 4 should be playable
        card_blue_4 = jnp.zeros((5, 5)).at[4, 3].set(1.0)
        assert aux.is_card_playable(card_blue_4, fireworks) == True
        status = aux.classify_card_status(card_blue_4, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 0
        
        # Any card from complete colors should be discardable
        card_red_1 = jnp.zeros((5, 5)).at[0, 0].set(1.0)
        assert aux.is_card_discardable(card_red_1, fireworks, deck, discard_pile, num_cards_of_rank) == True
    
    def test_prerequisite_check_with_rank_0(self):
        """Test that rank 0 (value 1) cards have no prerequisites to check."""
        card = jnp.zeros((5, 5)).at[1, 0].set(1.0)  # Yellow 1
        
        fireworks = jnp.zeros((5, 5))
        # Even if we set up weird discard pile state, rank 0 should not be affected
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        # Discard other Yellow cards
        discard_pile = discard_pile.at[0, 1, 1].set(1.0)
        discard_pile = discard_pile.at[0, 1, 2].set(1.0)
        
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # Yellow 1 should still be playable (no prerequisites)
        assert aux.is_card_playable(card, fireworks) == True
        assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank) == False
        
        status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
        assert status == 0
    
    def test_symmetric_colors(self):
        """Test that card logic is symmetric across all colors."""
        # Set up identical situations for all colors at rank 2
        fireworks = jnp.zeros((5, 5))
        for color in range(5):
            fireworks = fireworks.at[color, :2].set(1.0)  # All colors at rank 2
        
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        
        # All rank 3 (value 3) cards should have identical status
        statuses = []
        for color in range(5):
            card = jnp.zeros((5, 5)).at[color, 2].set(1.0)
            status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
            statuses.append(status)
        
        # All should be the same (playable = 0)
        assert all(s == 0 for s in statuses), f"All colors should have same status, got {statuses}"


class TestExtractHandStatus:
    """Test extract_hand_status function with actual Hanabi environment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = jax.random.PRNGKey(0)
        self.env = Hanabi(num_agents=2, num_colors=5, num_ranks=5, hand_size=5)
    
    def test_extract_hand_status_initial_state(self):
        """Test extract_hand_status on initial game state."""
        key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)
        
        # Create a batch of 4 environments
        num_envs = 4
        keys = jax.random.split(key, num_envs)
        
        def reset_env(k):
            _, s = self.env.reset(k)
            return s
        
        env_states = jax.vmap(reset_env)(keys)
        
        # Extract hand status
        hand_status, hand_slot_mask = aux.extract_hand_status(env_states)
        
        # Check shapes
        num_agents = 2
        hand_size = 5
        expected_batch_size = num_envs * num_agents
        
        assert hand_status.shape == (expected_batch_size, hand_size * 3)
        assert hand_slot_mask.shape == (expected_batch_size, hand_size)
        
        # All cards should be in hand initially (no empty slots)
        assert jnp.all(hand_slot_mask == 1.0)
        
        # Each card should have exactly one status (one-hot encoding)
        hand_status_reshaped = hand_status.reshape(expected_batch_size, hand_size, 3)
        assert jnp.allclose(hand_status_reshaped.sum(axis=2), 1.0)
    
    def test_extract_hand_status_after_play(self):
        """Test extract_hand_status after playing some cards."""
        key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)
        
        # Play a few moves
        for _ in range(5):
            # Get legal moves
            legal_moves = self.env.get_legal_moves(state)
            current_player = self.env.agents[jnp.argmax(state.cur_player_idx)]
            
            # Pick a random legal action
            key, subkey = jax.random.split(key)
            legal_actions = jnp.where(legal_moves[current_player] > 0)[0]
            action = jax.random.choice(subkey, legal_actions)
            
            # Step environment
            actions = {agent: action if agent == current_player else 0 for agent in self.env.agents}
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = self.env.step_env(subkey, state, actions)
            
            if dones["__all__"]:
                break
        
        # Create a batch with this state
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        
        # Extract hand status
        hand_status, hand_slot_mask = aux.extract_hand_status(env_states)
        
        # Check shapes
        num_agents = 2
        hand_size = 5
        expected_batch_size = 1 * num_agents
        
        assert hand_status.shape == (expected_batch_size, hand_size * 3)
        assert hand_slot_mask.shape == (expected_batch_size, hand_size)
        
        # Check that statuses are valid one-hot encodings
        hand_status_reshaped = hand_status.reshape(expected_batch_size, hand_size, 3)
        assert jnp.allclose(hand_status_reshaped.sum(axis=2), 1.0)
    
    def test_extract_hand_status_batched(self):
        """Test extract_hand_status with multiple environments."""
        key = self.key
        num_envs = 8
        
        # Create multiple environment states by looping (can't vmap because of dynamic shapes)
        env_states_list = []
        keys = jax.random.split(key, num_envs)
        
        for k in keys:
            _, s = self.env.reset(k)
            # Play a few random moves
            for i in range(3):
                legal_moves = self.env.get_legal_moves(s)
                current_player = self.env.agents[jnp.argmax(s.cur_player_idx)]
                k, subk = jax.random.split(k)
                legal_actions = jnp.where(legal_moves[current_player] > 0)[0]
                action = jax.random.choice(subk, legal_actions)
                actions = {agent: action if agent == current_player else 0 for agent in self.env.agents}
                k, subk = jax.random.split(k)
                _, s, _, dones, _ = self.env.step_env(subk, s, actions)
                if dones["__all__"]:
                    break
            env_states_list.append(s)
        
        # Stack the states into batched format
        env_states = jax.tree.map(lambda *args: jnp.stack(args), *env_states_list)
        
        # Extract hand status
        hand_status, hand_slot_mask = aux.extract_hand_status(env_states)
        
        # Check shapes
        num_agents = 2
        hand_size = 5
        expected_batch_size = num_envs * num_agents
        
        assert hand_status.shape == (expected_batch_size, hand_size * 3)
        assert hand_slot_mask.shape == (expected_batch_size, hand_size)
        
        # Check that all statuses are valid
        hand_status_reshaped = hand_status.reshape(expected_batch_size, hand_size, 3)
        assert jnp.allclose(hand_status_reshaped.sum(axis=2), 1.0)
        
        # Check that masks are binary
        assert jnp.all((hand_slot_mask == 0.0) | (hand_slot_mask == 1.0))
    
    def test_extract_hand_status_specific_scenario(self):
        """Test with a specific game scenario where we know card statuses."""
        # Create a specific deck where we know what should be playable
        colors = jnp.arange(5)
        ranks = jnp.arange(5)
        ranks = jnp.repeat(ranks, jnp.array([3, 2, 2, 2, 1]))
        color_rank_pairs = jnp.dstack(jnp.meshgrid(colors, ranks)).reshape(-1, 2)
        
        # Use a specific permutation for reproducibility
        key, subkey = jax.random.split(self.key)
        shuffled_pairs = jax.random.permutation(subkey, color_rank_pairs, axis=0)
        
        obs, state = self.env.reset_from_deck_of_pairs(shuffled_pairs)
        
        # Create batch
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        
        # Extract hand status
        hand_status, hand_slot_mask = aux.extract_hand_status(env_states)
        
        # At the start, all rank-0 cards should be playable
        # Let's check player 0's hand
        player0_status = hand_status[0].reshape(5, 3)
        player0_hand = state.player_hands[0]
        
        for i in range(5):
            card = player0_hand[i]
            color, rank = aux.get_card_color_and_rank(card)
            
            if rank == 0:  # Rank 0 cards are always playable at start
                # Should be classified as playable (status 0)
                assert player0_status[i, 0] == 1.0, f"Card {i} (rank {rank}) should be playable"
            # Other cards should be unknown (status 2) at the start
            # unless prerequisites are impossible (which shouldn't happen at start)
    
    def test_extract_hand_status_endgame_with_empty_slots(self):
        """Test extract_hand_status when deck is depleted and some hand slots are empty."""
        key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)
        
        # Play many moves to deplete the deck
        for _ in range(35):  # Play enough to deplete deck (50 cards - 10 in hands = 40 in deck)
            legal_moves = self.env.get_legal_moves(state)
            current_player = self.env.agents[jnp.argmax(state.cur_player_idx)]
            
            key, subkey = jax.random.split(key)
            legal_actions = jnp.where(legal_moves[current_player] > 0, size=20, fill_value=-1)[0]
            valid_actions = legal_actions[legal_actions >= 0]
            if len(valid_actions) == 0:
                break
            action = jax.random.choice(subkey, valid_actions)
            
            actions = {agent: action if agent == current_player else 0 for agent in self.env.agents}
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = self.env.step_env(subkey, state, actions)
            
            if dones["__all__"]:
                break
        
        # Create batch
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        
        # Extract hand status
        hand_status, hand_slot_mask = aux.extract_hand_status(env_states)
        
        # Check that masks correctly identify empty slots
        for player_idx in range(2):
            player_hand = state.player_hands[player_idx]
            player_mask = hand_slot_mask[player_idx]
            
            for card_idx in range(5):
                card = player_hand[card_idx]
                is_empty = ~card.any()
                expected_mask = 0.0 if is_empty else 1.0
                assert player_mask[card_idx] == expected_mask, \
                    f"Player {player_idx} card {card_idx}: mask mismatch"
    
    def test_extract_hand_status_all_rank_1_in_hand(self):
        """Test when a hand contains all rank 1 cards (all should be playable at start)."""
        # Create a deck with rank 1s at the front (enough for both players)
        deck_pairs = []
        
        # First 10 cards are all rank 1s (2 of each color)
        for i in range(10):
            deck_pairs.append([i % 5, 0])  # color, rank 0
        
        # Fill rest with other cards to make exactly 50 cards total
        # We need 40 more cards
        for color in range(5):
            for rank in range(5):
                copies = [3, 2, 2, 2, 1][rank]
                for _ in range(copies):
                    if len(deck_pairs) < 50:
                        # Skip the rank 0 cards we already added
                        if not (rank == 0 and len([p for p in deck_pairs if p[0] == color and p[1] == 0]) > 0):
                            deck_pairs.append([color, rank])
        
        # Ensure exactly 50 cards
        deck_pairs = jnp.array(deck_pairs[:50])
        
        obs, state = self.env.reset_from_deck_of_pairs(deck_pairs)
        
        # Create batch
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        
        # Extract hand status
        hand_status, hand_slot_mask = aux.extract_hand_status(env_states)
        
        # Both players should have rank 1 cards in their hands
        for player_idx in range(2):
            player_status = hand_status[player_idx].reshape(5, 3)
            player_hand = state.player_hands[player_idx]
            
            for card_idx in range(5):
                card = player_hand[card_idx]
                color, rank = aux.get_card_color_and_rank(card)
                if rank == 0:
                    # Rank 1 cards should be playable
                    assert player_status[card_idx, 0] == 1.0, \
                        f"Player {player_idx} card {card_idx} (rank 1) should be playable"
    
    def test_extract_hand_status_one_color_complete(self):
        """Test after completing one full color."""
        # Create a deterministic deck where Red cards come first
        deck_pairs = []
        # First, all Red cards in order (1-5)
        for rank in range(5):
            copies = [3, 2, 2, 2, 1][rank]
            for _ in range(copies):
                deck_pairs.append([0, rank])  # Red, rank
        
        # Then other colors
        for color in range(1, 5):
            for rank in range(5):
                copies = [3, 2, 2, 2, 1][rank]
                for _ in range(copies):
                    deck_pairs.append([color, rank])
        
        deck_pairs = jnp.array(deck_pairs)
        obs, state = self.env.reset_from_deck_of_pairs(deck_pairs)
        
        # Play the Red cards systematically
        key = self.key
        for move_count in range(15):  # Play enough to complete Red
            legal_moves = self.env.get_legal_moves(state)
            current_player = self.env.agents[jnp.argmax(state.cur_player_idx)]
            
            # Try to play red cards
            key, subkey = jax.random.split(key)
            legal_actions = jnp.where(legal_moves[current_player] > 0, size=20, fill_value=-1)[0]
            valid_actions = legal_actions[legal_actions >= 0]
            if len(valid_actions) == 0:
                break
            
            # Prefer play actions (5-9)
            play_actions = valid_actions[(valid_actions >= 5) & (valid_actions < 10)]
            if len(play_actions) > 0:
                action = play_actions[0]
            else:
                action = valid_actions[0]
            
            actions = {agent: action if agent == current_player else 0 for agent in self.env.agents}
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = self.env.step_env(subkey, state, actions)
            
            if dones["__all__"] or state.score >= 5:  # Red complete
                break
        
        # Create batch
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        
        # Extract hand status
        hand_status, hand_slot_mask = aux.extract_hand_status(env_states)
        
        # Any Red cards in hand should be discardable if Red is complete
        for player_idx in range(2):
            player_hand = state.player_hands[player_idx]
            player_status = hand_status[player_idx].reshape(5, 3)
            
            for card_idx in range(5):
                card = player_hand[card_idx]
                color, rank = aux.get_card_color_and_rank(card)
                
                if color == 0 and state.fireworks[0].sum() == 5:  # Red complete
                    # Red cards should be discardable
                    assert player_status[card_idx, 1] == 1.0, \
                        f"Red card should be discardable when Red is complete"
    
    def test_extract_hand_status_consistency_across_batch(self):
        """Test that the function produces consistent results for the same state."""
        key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)
        
        # Create two identical batches
        env_states1 = jax.tree.map(lambda x: jnp.stack([x, x]), state)
        env_states2 = jax.tree.map(lambda x: jnp.stack([x, x]), state)
        
        # Extract hand status
        hand_status1, hand_slot_mask1 = aux.extract_hand_status(env_states1)
        hand_status2, hand_slot_mask2 = aux.extract_hand_status(env_states2)
        
        # Results should be identical
        assert jnp.allclose(hand_status1, hand_status2)
        assert jnp.allclose(hand_slot_mask1, hand_slot_mask2)
        
        # First and second env in batch should be identical
        assert jnp.allclose(hand_status1[:2], hand_status1[2:4])
        assert jnp.allclose(hand_slot_mask1[:2], hand_slot_mask1[2:4])


class TestAdvancedLogic:
    """Advanced edge cases for card status classification."""

    def setup_method(self):
        self.fireworks = jnp.zeros((5, 5))
        self.deck = jnp.zeros((50, 5, 5))
        self.discard_pile = jnp.zeros((50, 5, 5))
        self.num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])

    def test_gap_in_impossible_chain(self):
        """Test when a gap exists in the impossibility chain (Rank 3 impossible, Rank 2 possible, checking Rank 5)."""
        # Firework at 0 (nothing played)
        # We hold Rank 4 (Yellow 5)
        card = jnp.zeros((5, 5)).at[1, 4].set(1.0)
        
        # Rank 0 (Yellow 1) is possible (default)
        # Rank 1 (Yellow 2) is possible (default)
        # Rank 2 (Yellow 3) is IMPOSSIBLE (all 2 copies discarded)
        discard_pile = self.discard_pile.at[0, 1, 2].set(1.0)
        discard_pile = discard_pile.at[1, 1, 2].set(1.0)
        
        # Rank 3 (Yellow 4) is possible (default)
        
        # Since Rank 2 is impossible, Rank 4 should be discardable
        assert aux.is_card_discardable(card, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        status = aux.classify_card_status(card, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        assert status == 1  # Discardable

    def test_last_copy_in_hand_is_not_discardable(self):
        """Test that if the last copy of a prereq is in hand (implied), the target card is not discardable."""
        # Firework 0
        # We check Rank 1 (Red 2)
        card = jnp.zeros((5, 5)).at[0, 1].set(1.0)
        
        # Rank 0 (Red 1) - 3 copies total.
        # 2 copies are discarded.
        discard_pile = self.discard_pile.at[0, 0, 0].set(1.0)
        discard_pile = discard_pile.at[1, 0, 0].set(1.0)
        # 1 copy remains (implicitly in deck or hand).
        
        # Rank 1 should NOT be discardable, because Rank 0 is still possible.
        assert not aux.is_card_discardable(card, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        status = aux.classify_card_status(card, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        assert status == 2  # Unknown (waiting for Rank 0)

    def test_completed_color_rank_5_discardable(self):
        """Test that even the highest rank card is discardable if the color is complete."""
        # Blue complete (5)
        fireworks = self.fireworks.at[4, :].set(1.0)
        
        # Check Blue 5 (Rank 4)
        card = jnp.zeros((5, 5)).at[4, 4].set(1.0)
        
        # Should be discardable because rank (4) < level (5)
        assert aux.is_card_discardable(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
        assert not aux.is_card_playable(card, fireworks)
        status = aux.classify_card_status(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
        assert status == 1 # Discardable

    def test_card_matches_current_firework_level_playable(self):
        """Test exact match logic: Card Rank == Firework Level => Playable."""
        # Green at Level 2 (Ranks 0, 1 played)
        fireworks = self.fireworks.at[2, 0].set(1.0)
        fireworks = fireworks.at[2, 1].set(1.0)
        
        # Check Green 3 (Rank 2)
        card = jnp.zeros((5, 5)).at[2, 2].set(1.0)
        
        assert aux.is_card_playable(card, fireworks)
        assert not aux.is_card_discardable(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
        status = aux.classify_card_status(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
        assert status == 0 # Playable

    def test_card_below_current_firework_level_discardable(self):
        """Test exact match logic: Card Rank < Firework Level => Discardable."""
        # Green at Level 2 (Ranks 0, 1 played)
        fireworks = self.fireworks.at[2, 0].set(1.0)
        fireworks = fireworks.at[2, 1].set(1.0)
        
        # Check Green 2 (Rank 1)
        card = jnp.zeros((5, 5)).at[2, 1].set(1.0)
        
        assert not aux.is_card_playable(card, fireworks)
        assert aux.is_card_discardable(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
        status = aux.classify_card_status(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
        assert status == 1 # Discardable

    def test_card_above_current_firework_level_unknown(self):
        """Test exact match logic: Card Rank > Firework Level => Unknown (if prereqs exist)."""
        # Green at Level 2 (Ranks 0, 1 played)
        fireworks = self.fireworks.at[2, 0].set(1.0)
        fireworks = fireworks.at[2, 1].set(1.0)
        
        # Check Green 4 (Rank 3)
        card = jnp.zeros((5, 5)).at[2, 3].set(1.0)
        
        # Assuming Rank 2 is possible
        assert not aux.is_card_playable(card, fireworks)
        assert not aux.is_card_discardable(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
        status = aux.classify_card_status(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
        assert status == 2 # Unknown

    def test_multiple_impossible_prereqs(self):
        """Test that multiple impossible prerequisites don't confuse the logic."""
        # Firework 0
        # Check Rank 4
        card = jnp.zeros((5, 5)).at[3, 4].set(1.0)
        
        # Rank 0 impossible
        discard_pile = self.discard_pile
        for i in range(3):
            discard_pile = discard_pile.at[i, 3, 0].set(1.0)
            
        # Rank 2 impossible
        for i in range(2):
            discard_pile = discard_pile.at[i, 3, 2].set(1.0)
            
        assert aux.is_card_discardable(card, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        status = aux.classify_card_status(card, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        assert status == 1


class TestHyperTrickyEdgeCases:
    """Extremely tricky edge cases that stress-test the card status logic."""

    def setup_method(self):
        self.fireworks = jnp.zeros((5, 5))
        self.deck = jnp.zeros((50, 5, 5))
        self.discard_pile = jnp.zeros((50, 5, 5))
        self.num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])

    def test_transitive_impossibility_chain(self):
        """Test that impossibility propagates through rank chains."""
        # If rank 1 is impossible, then ranks 2,3,4,5 should all be discardable
        # NOTE: The implementation doesn't check if the card itself is impossible,
        # only if prerequisites are impossible. So Red 1 will still be classified as playable
        # even if all copies are discarded (this is a limitation of the current implementation).

        # Make all Red 1s impossible (discard all 3 copies)
        discard_pile = self.discard_pile.at[0, 0, 0].set(1.0)  # Red 1
        discard_pile = discard_pile.at[1, 0, 0].set(1.0)  # Red 1
        discard_pile = discard_pile.at[2, 0, 0].set(1.0)  # Red 1

        # Red 1 is still classified as "playable" by the current implementation
        # because rank (0) == firework_level (0), even though all copies are gone
        card_r1 = jnp.zeros((5, 5)).at[0, 0].set(1.0)
        # The implementation will say it's playable (doesn't check if card itself is impossible)
        assert aux.is_card_playable(card_r1, self.fireworks)
        # It's not discardable because rank (0) is not < firework_level (0)
        assert not aux.is_card_discardable(card_r1, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        # So it will be classified as playable (status 0)
        assert aux.classify_card_status(card_r1, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank) == 0

        # All higher ranks should be discardable due to impossible prerequisite (Red 1)
        for rank in range(1, 5):  # ranks 2,3,4,5
            card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
            assert aux.is_card_discardable(card, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank), \
                f"Red {rank+1} should be discardable when Red 1 is impossible"
            assert aux.classify_card_status(card, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank) == 1

    def test_mixed_prereq_availability(self):
        """Test cards where prerequisites are partially available (some played, some discarded)."""
        # Red 1: 1 played, 2 discarded -> all 3 copies gone
        # LIMITATION: The implementation doesn't detect this case because Red 1 is already
        # on the fireworks (level=1), so when checking Red 2 (rank=1), there are no
        # prerequisites to check between level (1) and rank (1).
        fireworks = self.fireworks.at[0, 0].set(1.0)  # Red 1 played
        discard_pile = self.discard_pile.at[0, 0, 0].set(1.0)  # Red 1 discarded
        discard_pile = discard_pile.at[1, 0, 0].set(1.0)  # Red 1 discarded

        card_r2 = jnp.zeros((5, 5)).at[0, 1].set(1.0)  # Red 2
        # Red 2 will be classified as playable (not discardable) because rank (1) == level (1)
        # even though we might want it to be discardable since all Red 1s are gone
        assert aux.is_card_playable(card_r2, fireworks)
        assert not aux.is_card_discardable(card_r2, fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        assert aux.classify_card_status(card_r2, fireworks, self.deck, discard_pile, self.num_cards_of_rank) == 0

    def test_partial_prereq_loss_blocks_discard(self):
        """Test that cards are NOT discardable if only some prerequisites are lost."""
        # Red 1: only 2 out of 3 discarded (1 still possibly available)
        # Red 2 should NOT be discardable
        discard_pile = self.discard_pile.at[0, 0, 0].set(1.0)  # Red 1 discarded
        discard_pile = discard_pile.at[1, 0, 0].set(1.0)  # Red 1 discarded
        # One Red 1 still potentially available

        card_r2 = jnp.zeros((5, 5)).at[0, 1].set(1.0)  # Red 2
        assert not aux.is_card_discardable(card_r2, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        assert aux.classify_card_status(card_r2, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank) == 2  # unknown

    def test_rank_5_with_broken_chain(self):
        """Test rank 5 cards when intermediate ranks become impossible."""
        # Make Red 2 impossible, then Red 5 should be discardable
        discard_pile = self.discard_pile.at[0, 0, 1].set(1.0)  # Red 2 discarded
        discard_pile = discard_pile.at[1, 0, 1].set(1.0)  # Red 2 discarded

        card_r5 = jnp.zeros((5, 5)).at[0, 4].set(1.0)  # Red 5
        assert aux.is_card_discardable(card_r5, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        assert aux.classify_card_status(card_r5, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank) == 1

    def test_critical_last_copy_protection(self):
        """Test that the last copy of a critical rank 5 card is not discardable."""
        # Only 1 copy of each rank 5 exists. If it's playable, it should NOT be discardable
        fireworks = self.fireworks
        # Complete Red up to rank 4
        fireworks = fireworks.at[0, :4].set(1.0)  # Red 1-4 played

        card_r5 = jnp.zeros((5, 5)).at[0, 4].set(1.0)  # Red 5 (the ONLY copy)
        # Should be playable, not discardable
        assert aux.is_card_playable(card_r5, fireworks)
        assert not aux.is_card_discardable(card_r5, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
        assert aux.classify_card_status(card_r5, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank) == 0

    def test_card_below_firework_level_discardable(self):
        """Test that cards below current firework level are discardable."""
        fireworks = self.fireworks.at[0, :3].set(1.0)  # Red 1-3 played

        # Red 1, Red 2 should be discardable (below level 3)
        for rank in range(3):
            card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
            assert aux.is_card_discardable(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank), \
                f"Red {rank+1} should be discardable when Red 1-3 are played"
            assert aux.classify_card_status(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank) == 1

    def test_card_at_firework_level_playable(self):
        """Test that cards exactly at firework level are playable."""
        fireworks = self.fireworks.at[0, :2].set(1.0)  # Red 1-2 played

        card_r3 = jnp.zeros((5, 5)).at[0, 2].set(1.0)  # Red 3 (level = 2, so next needed is rank 2)
        assert aux.is_card_playable(card_r3, fireworks)
        assert not aux.is_card_discardable(card_r3, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
        assert aux.classify_card_status(card_r3, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank) == 0

    def test_complex_multi_rank_impossibility(self):
        """Test when multiple ranks in the same color become impossible."""
        # Make both Red 2 and Red 3 impossible
        discard_pile = self.discard_pile.at[0, 0, 1].set(1.0)  # Red 2
        discard_pile = discard_pile.at[1, 0, 1].set(1.0)  # Red 2
        discard_pile = discard_pile.at[0, 0, 2].set(1.0)  # Red 3
        discard_pile = discard_pile.at[1, 0, 2].set(1.0)  # Red 3

        # Red 4 and Red 5 should be discardable
        for rank in [3, 4]:  # ranks 4 and 5
            card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
            assert aux.is_card_discardable(card, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank), \
                f"Red {rank+1} should be discardable when Red 2 and 3 are impossible"

    def test_cross_color_independence(self):
        """Test that impossibility in one color doesn't affect other colors."""
        # Make all Red cards impossible by discarding all Red 1s
        discard_pile = self.discard_pile
        for i in range(3):
            discard_pile = discard_pile.at[i, 0, 0].set(1.0)  # All Red 1s discarded

        # Yellow cards should be unaffected - Yellow 1 should still be playable
        card_y1 = jnp.zeros((5, 5)).at[1, 0].set(1.0)  # Yellow 1
        assert aux.is_card_playable(card_y1, self.fireworks)
        assert not aux.is_card_discardable(card_y1, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank)

    def test_rank_5_card_with_prereq_played(self):
        """Test rank 5 when prerequisite (rank 4) is played but rank 5 is still needed."""
        fireworks = self.fireworks.at[0, :4].set(1.0)  # Red 1-4 played

        card_r5 = jnp.zeros((5, 5)).at[0, 4].set(1.0)  # Red 5
        # Should be playable (rank 4 == current level 4)
        assert aux.is_card_playable(card_r5, fireworks)
        assert not aux.is_card_discardable(card_r5, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)


    def test_empty_card_handling(self):
        """Test that empty cards are handled correctly in all functions."""
        empty_card = jnp.zeros((5, 5))

        # Empty cards should return -1 for color/rank
        color, rank = aux.get_card_color_and_rank(empty_card)
        assert color == -1 and rank == -1

        # Empty cards should not be playable or discardable
        assert not aux.is_card_playable(empty_card, self.fireworks)
        assert not aux.is_card_discardable(empty_card, self.fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)

        # Empty cards should be classified as unknown (2)
        status = aux.classify_card_status(empty_card, self.fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
        assert status == 2

    def test_all_rank_1_cards_at_start(self):
        """Test that all rank 1 cards are playable at the start of the game."""
        # At game start, all fireworks are empty
        for color in range(5):
            card = jnp.zeros((5, 5)).at[color, 0].set(1.0)  # Rank 1 for each color
            assert aux.is_card_playable(card, self.fireworks), f"Rank 1 card for color {color} should be playable"
            assert not aux.is_card_discardable(card, self.fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
            status = aux.classify_card_status(card, self.fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
            assert status == 0, f"Rank 1 card for color {color} should be classified as playable"

    def test_rank_5_edge_case_single_copy(self):
        """Test the edge case of rank 5 cards (only 1 copy exists)."""
        # Rank 5 cards have only 1 copy, so they're critical when they're the last one
        fireworks = self.fireworks.at[0, :4].set(1.0)  # Red 1-4 played

        # The last Red 5 should be playable
        card_r5 = jnp.zeros((5, 5)).at[0, 4].set(1.0)
        assert aux.is_card_playable(card_r5, fireworks)

        # But if we somehow have no Red 5 left, higher cards don't exist
        # (This is more of a sanity check)

    def test_impossible_prereq_vs_obsolete_card(self):
        """Test the difference between impossible prerequisites vs already played cards."""
        # Set up: Red 1-2 played, Red 3 discarded (but Red 3 is still possible since we have it on fireworks)
        fireworks = self.fireworks.at[0, :2].set(1.0)  # Red 1-2 played

        # Red 1 and Red 2 are obsolete (below firework level)
        card_r1 = jnp.zeros((5, 5)).at[0, 0].set(1.0)
        assert aux.is_card_discardable(card_r1, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)

        card_r2 = jnp.zeros((5, 5)).at[0, 1].set(1.0)
        assert aux.is_card_discardable(card_r2, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)

        # Red 3 is playable (at firework level)
        card_r3 = jnp.zeros((5, 5)).at[0, 2].set(1.0)
        assert aux.is_card_playable(card_r3, fireworks)
        assert not aux.is_card_discardable(card_r3, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)

    def test_boundary_between_playable_and_discardable(self):
        """Test the exact boundary where cards transition from playable to discardable."""
        # When firework level = N, rank N is playable, rank < N is discardable
        for level in range(5):
            fireworks = jnp.zeros((5, 5))
            if level > 0:
                fireworks = fireworks.at[0, :level].set(1.0)  # Red at level
            
            for rank in range(5):
                card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
                
                if rank < level:
                    # Below level: discardable
                    assert aux.is_card_discardable(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank), \
                        f"Rank {rank} should be discardable when level is {level}"
                    assert not aux.is_card_playable(card, fireworks)
                    assert aux.classify_card_status(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank) == 1
                elif rank == level:
                    # At level: playable
                    assert aux.is_card_playable(card, fireworks), \
                        f"Rank {rank} should be playable when level is {level}"
                    assert not aux.is_card_discardable(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
                    assert aux.classify_card_status(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank) == 0
                else:
                    # Above level: unknown (if prerequisites possible)
                    assert not aux.is_card_playable(card, fireworks)
                    assert not aux.is_card_discardable(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank)
                    assert aux.classify_card_status(card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank) == 2

    def test_prerequisite_counting_accuracy(self):
        """Test that prerequisite counting is accurate (played + discarded vs total)."""
        # Test with rank 2 cards (2 copies exist)
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, 0].set(1.0)  # Red 1 played, level = 1
        
        discard_pile = jnp.zeros((50, 5, 5))
        
        # Case 1: 0 Red 2s discarded (2 remain) - Red 3 should be unknown
        card_r3 = jnp.zeros((5, 5)).at[0, 2].set(1.0)
        # Checking Red 3 (rank=2) with level=1: checks prerequisite rank 1 (Red 2)
        # 0 played + 0 discarded < 2 total → Red 2 is possible → Red 3 is unknown
        assert not aux.is_card_discardable(card_r3, fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        
        # Case 2: 1 Red 2 discarded (1 remains) - Red 3 should still be unknown
        discard_pile = discard_pile.at[0, 0, 1].set(1.0)
        # 0 played + 1 discarded < 2 total → Red 2 is still possible → Red 3 is unknown
        assert not aux.is_card_discardable(card_r3, fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        
        # Case 3: 2 Red 2s discarded (0 remain) - Red 3 should be discardable
        discard_pile = discard_pile.at[1, 0, 1].set(1.0)
        # 0 played + 2 discarded >= 2 total → Red 2 is impossible → Red 3 is discardable
        assert aux.is_card_discardable(card_r3, fireworks, self.deck, discard_pile, self.num_cards_of_rank)
        
        # Case 4: 1 Red 2 played + 1 discarded (0 remain)
        # LIMITATION: When Red 2 is played, level becomes 2, so checking Red 3 (rank=2)
        # means checking prerequisites between 2 and 2 → none! So Red 3 is playable, not discardable
        fireworks2 = fireworks.at[0, 1].set(1.0)  # Red 2 played, level = 2
        discard_pile2 = jnp.zeros((50, 5, 5)).at[0, 0, 1].set(1.0)  # 1 Red 2 discarded
        # Red 3 (rank=2) with level=2 → playable, not discardable
        assert not aux.is_card_discardable(card_r3, fireworks2, self.deck, discard_pile2, self.num_cards_of_rank)
        assert aux.is_card_playable(card_r3, fireworks2)


class TestJAXCompatibility:
    """Test that all functions are JAX-compatible (JIT, vmap, grad)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.card = jnp.zeros((5, 5)).at[0, 0].set(1.0)
        self.fireworks = jnp.zeros((5, 5))
        self.deck = jnp.zeros((50, 5, 5))
        self.discard_pile = jnp.zeros((50, 5, 5))
        self.num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
    
    def test_jit_compatibility(self):
        """Test that functions can be JIT compiled."""
        jitted_get_card = jax.jit(aux.get_card_color_and_rank)
        jitted_playable = jax.jit(aux.is_card_playable)
        jitted_discardable = jax.jit(aux.is_card_discardable)
        jitted_classify = jax.jit(aux.classify_card_status)
        
        # Run jitted functions
        color, rank = jitted_get_card(self.card)
        is_playable = jitted_playable(self.card, self.fireworks)
        is_discardable = jitted_discardable(
            self.card, self.fireworks, self.deck, self.discard_pile, self.num_cards_of_rank
        )
        status = jitted_classify(
            self.card, self.fireworks, self.deck, self.discard_pile, self.num_cards_of_rank
        )
        
        # Just check they run without error
        assert True
    
    def test_vmap_compatibility(self):
        """Test that functions work with vmap."""
        # Create batch of cards
        cards = jnp.stack([
            jnp.zeros((5, 5)).at[0, 0].set(1.0),
            jnp.zeros((5, 5)).at[1, 1].set(1.0),
            jnp.zeros((5, 5)).at[2, 2].set(1.0),
        ])
        
        # vmap over cards
        colors, ranks = jax.vmap(aux.get_card_color_and_rank)(cards)
        assert colors.shape == (3,)
        assert ranks.shape == (3,)
        
        is_playable = jax.vmap(lambda c: aux.is_card_playable(c, self.fireworks))(cards)
        assert is_playable.shape == (3,)


class TestRealHanabiIntegration:
    """Test card status functions with actual Hanabi game states."""

    def setup_method(self):
        self.key = jax.random.PRNGKey(42)
        self.env = Hanabi(num_agents=2, num_colors=5, num_ranks=5, hand_size=5)

    def test_initial_game_state_all_rank_1_playable(self):
        """Test that in a fresh game, rank 1 cards are correctly identified as playable."""
        key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)

        # Get hand status
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        hand_status, hand_slot_mask = aux.extract_hand_status(env_states)

        # All cards should be present (no empty slots)
        assert jnp.all(hand_slot_mask == 1.0)

        # Reshape to (num_agents, hand_size, 3)
        hand_status_reshaped = hand_status.reshape(2, 5, 3)

        # Check each player's hand
        for player_idx in range(2):
            player_hand = state.player_hands[player_idx]
            player_status = hand_status_reshaped[player_idx]

            for card_idx in range(5):
                card = player_hand[card_idx]
                color, rank = aux.get_card_color_and_rank(card)
                status_one_hot = player_status[card_idx]

                # At game start, rank 0 (value 1) cards should be playable
                if rank == 0:
                    assert jnp.allclose(status_one_hot, jnp.array([1.0, 0.0, 0.0])), \
                        f"Rank 1 card {card_idx} for player {player_idx} should be playable"
                # Other ranks should be unknown (waiting for prerequisites)
                else:
                    assert jnp.allclose(status_one_hot, jnp.array([0.0, 0.0, 1.0])), \
                        f"Rank {rank+1} card {card_idx} for player {player_idx} should be unknown"

    def test_after_playing_rank_1_cards(self):
        """Test status after playing some rank 1 cards."""
        key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)

        # Find and play a Red 1 card
        player_hand = state.player_hands[jnp.argmax(state.cur_player_idx)]  # Current player
        red_1_action = None
        for card_idx in range(5):
            card = player_hand[card_idx]
            color, rank = aux.get_card_color_and_rank(card)
            if color == 0 and rank == 0:  # Red 1
                red_1_action = 5 + card_idx  # Play action (play actions are 5-9)
                break

        if red_1_action is not None:
            # Play the Red 1
            current_player = self.env.agents[jnp.argmax(state.cur_player_idx)]
            actions = {agent: red_1_action if agent == current_player else 20 for agent in self.env.agents}  # 20 is noop
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = self.env.step_env(subkey, state, actions)

            # Check status now - Red firework should be at level 1
            assert state.fireworks[0].sum() == 1, "Red firework should be at level 1"

            env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
            hand_status, hand_slot_mask = aux.extract_hand_status(env_states)
            hand_status_reshaped = hand_status.reshape(2, 5, 3)

            # Check all players' hands
            for player_idx in range(2):
                player_hand = state.player_hands[player_idx]
                player_status = hand_status_reshaped[player_idx]

                for card_idx in range(5):
                    card = player_hand[card_idx]
                    if card.any():  # Not empty
                        color, rank = aux.get_card_color_and_rank(card)
                        status_one_hot = player_status[card_idx]

                        if color == 0:  # Red cards
                            if rank == 0:  # Red 1s
                                # Should be discardable (below firework level 1)
                                assert jnp.allclose(status_one_hot, jnp.array([0.0, 1.0, 0.0])), \
                                    f"Red 1 cards should be discardable after one is played"
                            elif rank == 1:  # Red 2s
                                # Should be playable (at firework level 1)
                                assert jnp.allclose(status_one_hot, jnp.array([1.0, 0.0, 0.0])), \
                                    f"Red 2 cards should be playable after Red 1 is played"

    def test_completed_color_cards_become_discardable(self):
        """Test that cards of a completed color become discardable."""
        # Create a custom deck where Red cards come first for easy completion
        deck_pairs = []
        # Add all Red cards in order
        for rank in range(5):
            copies = [3, 2, 2, 2, 1][rank]
            for _ in range(copies):
                deck_pairs.append([0, rank])  # Red, rank

        # Fill rest with other cards
        for color in range(1, 5):
            for rank in range(5):
                copies = [3, 2, 2, 2, 1][rank]
                for _ in range(copies):
                    deck_pairs.append([color, rank])

        deck_pairs = jnp.array(deck_pairs[:50])  # Ensure 50 cards
        obs, state = self.env.reset_from_deck_of_pairs(deck_pairs)

        # Play enough moves to complete Red
        key = self.key
        for _ in range(20):  # Should be enough to complete Red
            legal_moves = self.env.get_legal_moves(state)
            current_player = self.env.agents[jnp.argmax(state.cur_player_idx)]

            # Try to play Red cards preferentially
            actions = {agent: 0 for agent in self.env.agents}  # Default to noop-ish

            if legal_moves[current_player].any():
                # Look for play actions (5-9) that might be Red
                for action in range(5, 10):  # Play actions
                    if legal_moves[current_player][action]:
                        actions[current_player] = action
                        break

                # If no play action found, take any legal action
                if actions[current_player] == 0:
                    legal_actions = jnp.where(legal_moves[current_player])[0]
                    actions[current_player] = legal_actions[0]

            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = self.env.step_env(subkey, state, actions)

            # Check if Red is complete
            if state.fireworks[0].sum() == 5:
                break

        # Now check that Red cards are discardable
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        hand_status, hand_slot_mask = aux.extract_hand_status(env_states)
        hand_status_reshaped = hand_status.reshape(2, 5, 3)

        for player_idx in range(2):
            player_hand = state.player_hands[player_idx]
            player_status = hand_status_reshaped[player_idx]

            for card_idx in range(5):
                card = player_hand[card_idx]
                if card.any():
                    color, rank = aux.get_card_color_and_rank(card)
                    status_one_hot = player_status[card_idx]

                    if color == 0:  # Red cards
                        # Should be discardable since Red is complete
                        assert jnp.allclose(status_one_hot, jnp.array([0.0, 1.0, 0.0])), \
                            f"Red card (rank {rank}) should be discardable when Red is complete"

    def test_impossible_card_detection(self):
        """Test detection of cards that become impossible due to discarded prerequisites."""
        # Create a scenario where some rank becomes impossible
        key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)

        # Manually modify state to discard all copies of Red 2
        # This is tricky to do naturally, so we'll create a synthetic test

        # Instead, let's just test the logic with known states
        fireworks = jnp.zeros((5, 5))
        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))

        # Discard all Red 2s (rank 1)
        discard_pile = discard_pile.at[0, 0, 1].set(1.0)
        discard_pile = discard_pile.at[1, 0, 1].set(1.0)

        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])

        # Red 3 should be discardable
        card_r3 = jnp.zeros((5, 5)).at[0, 2].set(1.0)
        assert aux.is_card_discardable(card_r3, fireworks, deck, discard_pile, num_cards_of_rank)

        # Red 4 and Red 5 should also be discardable
        for rank in [3, 4]:
            card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
            assert aux.is_card_discardable(card, fireworks, deck, discard_pile, num_cards_of_rank), \
                f"Red {rank+1} should be discardable when Red 2 is impossible"

    def test_mixed_scenario_complex_state(self):
        """Test a complex mixed scenario with multiple colors at different stages."""
        # Set up a complex firework state
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :5].set(1.0)  # Red complete
        fireworks = fireworks.at[1, :3].set(1.0)  # Yellow at 3
        fireworks = fireworks.at[2, :1].set(1.0)  # Green at 1
        # Blue and White at 0

        deck = jnp.zeros((50, 5, 5))
        discard_pile = jnp.zeros((50, 5, 5))
        num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])

        # Test various cards
        test_cases = [
            # (color, rank, expected_status: 0=playable, 1=discardable, 2=unknown)
            (0, 0, 1),  # Red 1: discardable (Red complete)
            (0, 4, 1),  # Red 5: discardable (Red complete)
            (1, 0, 1),  # Yellow 1: discardable (below Yellow level 3)
            (1, 1, 1),  # Yellow 2: discardable (below Yellow level 3)
            (1, 2, 1),  # Yellow 3: discardable (below Yellow level 3)
            (1, 3, 0),  # Yellow 4: playable (at Yellow level 3)
            (2, 0, 1),  # Green 1: discardable (below Green level 1)
            (2, 1, 0),  # Green 2: playable (at Green level 1)
            (2, 2, 2),  # Green 3: unknown (above Green level 1)
            (3, 0, 0),  # Blue 1: playable (Blue at 0)
            (4, 0, 0),  # White 1: playable (White at 0)
        ]

        for color, rank, expected_status in test_cases:
            card = jnp.zeros((5, 5)).at[color, rank].set(1.0)
            actual_status = aux.classify_card_status(card, fireworks, deck, discard_pile, num_cards_of_rank)
            assert actual_status == expected_status, \
                f"Card (color {color}, rank {rank}) should be status {expected_status}, got {actual_status}"


class TestGlobalProperties:
    """Property-style tests over many Hanabi-like states."""

    def setup_method(self):
        self.num_colors = 5
        self.num_ranks = 5
        # Standard Hanabi distribution used in JaxMARL
        self.num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])
        self.deck_size = 50
        self.rng = np.random.default_rng(0)

    def _random_fireworks(self):
        """Sample fireworks consistent with HanabiGame: prefix-of-ones per color."""
        fireworks = jnp.zeros((self.num_colors, self.num_ranks))
        levels = self.rng.integers(0, self.num_ranks + 1, size=self.num_colors)
        for c in range(self.num_colors):
            level = int(levels[c])
            if level > 0:
                fireworks = fireworks.at[c, :level].set(1.0)
        return fireworks

    def _random_discard_pile(self, fireworks):
        """Sample a discard pile consistent with card counts and current fireworks."""
        discard_pile = jnp.zeros((self.deck_size, self.num_colors, self.num_ranks))
        # For each (color, rank), choose 0..remaining copies to discard
        for c in range(self.num_colors):
            for r in range(self.num_ranks):
                already_on_fireworks = int(fireworks[c, r])
                max_copies = int(self.num_cards_of_rank[r] - already_on_fireworks)
                if max_copies <= 0:
                    continue
                n_discard = int(self.rng.integers(0, max_copies + 1))
                if n_discard > 0:
                    # Mark first n_discard slots for this (color, rank) as discarded.
                    discard_pile = discard_pile.at[:n_discard, c, r].set(1.0)
        return discard_pile

    def test_playable_and_discardable_are_mutually_exclusive(self):
        """No card should ever be both playable and discardable in any sane state."""
        deck = jnp.zeros((self.deck_size, self.num_colors, self.num_ranks))

        for _ in range(20):
            fireworks = self._random_fireworks()
            discard_pile = self._random_discard_pile(fireworks)

            for c in range(self.num_colors):
                for r in range(self.num_ranks):
                    card = jnp.zeros((self.num_colors, self.num_ranks)).at[c, r].set(1.0)
                    playable = bool(aux.is_card_playable(card, fireworks))
                    discardable = bool(
                        aux.is_card_discardable(
                            card, fireworks, deck, discard_pile, self.num_cards_of_rank
                        )
                    )
                    assert not (
                        playable and discardable
                    ), f"Card (color={c}, rank={r}) cannot be both playable and discardable."

    def test_classify_matches_playable_and_discardable(self):
        """classify_card_status should agree with is_card_playable / is_card_discardable."""
        deck = jnp.zeros((self.deck_size, self.num_colors, self.num_ranks))

        for _ in range(20):
            fireworks = self._random_fireworks()
            discard_pile = self._random_discard_pile(fireworks)

            for c in range(self.num_colors):
                for r in range(self.num_ranks):
                    card = jnp.zeros((self.num_colors, self.num_ranks)).at[c, r].set(1.0)
                    is_playable = bool(aux.is_card_playable(card, fireworks))
                    is_discardable = bool(
                        aux.is_card_discardable(
                            card, fireworks, deck, discard_pile, self.num_cards_of_rank
                        )
                    )
                    status = int(
                        aux.classify_card_status(
                            card, fireworks, deck, discard_pile, self.num_cards_of_rank
                        )
                    )

                    if is_playable:
                        assert status == 0, "Playable card must be classified as status 0."
                    elif is_discardable:
                        assert status == 1, "Discardable card must be classified as status 1."
                    else:
                        assert status == 2, "Non-playable, non-discardable card must be status 2."


class TestOriginalLogic:
    """Test the original C++ logic (simple rank comparison with firework level)."""

    def setup_method(self):
        self.fireworks = jnp.zeros((5, 5))

    def test_original_playable_rank_0_empty_fireworks(self):
        """Test that rank 0 cards are playable on empty fireworks."""
        card = jnp.zeros((5, 5)).at[0, 0].set(1.0)  # Red 1
        assert aux.classify_card_status_original(card, self.fireworks) == 0  # playable

    def test_original_playable_rank_1_after_rank_0(self):
        """Test that rank 1 is playable when rank 0 is on fireworks."""
        fireworks = self.fireworks.at[0, 0].set(1.0)  # Red 1 played
        card = jnp.zeros((5, 5)).at[0, 1].set(1.0)  # Red 2
        assert aux.classify_card_status_original(card, fireworks) == 0  # playable

    def test_original_discardable_below_firework_level(self):
        """Test that cards below firework level are discardable."""
        fireworks = self.fireworks.at[0, 0].set(1.0)  # Red 1 played
        fireworks = fireworks.at[0, 1].set(1.0)  # Red 2 played
        card = jnp.zeros((5, 5)).at[0, 0].set(1.0)  # Red 1
        assert aux.classify_card_status_original(card, fireworks) == 1  # discardable

    def test_original_unknown_above_firework_level(self):
        """Test that cards above firework level are unknown."""
        fireworks = self.fireworks.at[0, 0].set(1.0)  # Red 1 played
        card = jnp.zeros((5, 5)).at[0, 2].set(1.0)  # Red 3
        assert aux.classify_card_status_original(card, fireworks) == 2  # unknown

    def test_original_empty_card_is_unknown(self):
        """Test that empty cards are classified as unknown."""
        card = jnp.zeros((5, 5))
        assert aux.classify_card_status_original(card, self.fireworks) == 2  # unknown

    def test_original_all_ranks_at_start(self):
        """Test all rank classifications at game start."""
        # At start, fireworks are empty (level 0 for all colors)
        for color in range(5):
            # Rank 0 should be playable
            card_r0 = jnp.zeros((5, 5)).at[color, 0].set(1.0)
            assert aux.classify_card_status_original(card_r0, self.fireworks) == 0

            # All other ranks should be unknown
            for rank in range(1, 5):
                card = jnp.zeros((5, 5)).at[color, rank].set(1.0)
                assert aux.classify_card_status_original(card, self.fireworks) == 2

    def test_original_complete_color_all_discardable(self):
        """Test that all cards are discardable when a color is complete."""
        fireworks = self.fireworks.at[0, :].set(1.0)  # Red complete
        for rank in range(5):
            card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
            assert aux.classify_card_status_original(card, fireworks) == 1  # discardable

    def test_original_multiple_colors_independent(self):
        """Test that colors are independent in classification."""
        fireworks = self.fireworks.at[0, :3].set(1.0)  # Red at level 3
        fireworks = fireworks.at[1, 0].set(1.0)         # Yellow at level 1
        
        # Red rank 3 should be playable
        assert aux.classify_card_status_original(
            jnp.zeros((5, 5)).at[0, 3].set(1.0), fireworks
        ) == 0
        
        # Red rank 2 should be discardable
        assert aux.classify_card_status_original(
            jnp.zeros((5, 5)).at[0, 2].set(1.0), fireworks
        ) == 1
        
        # Yellow rank 1 should be playable
        assert aux.classify_card_status_original(
            jnp.zeros((5, 5)).at[1, 1].set(1.0), fireworks
        ) == 0
        
        # Yellow rank 2 should be unknown
        assert aux.classify_card_status_original(
            jnp.zeros((5, 5)).at[1, 2].set(1.0), fireworks
        ) == 2

    def test_original_rank_5_playable_when_ready(self):
        """Test that rank 5 is playable when rank 4 is complete."""
        fireworks = self.fireworks.at[0, :4].set(1.0)  # Red 1-4 played
        card = jnp.zeros((5, 5)).at[0, 4].set(1.0)  # Red 5
        assert aux.classify_card_status_original(card, fireworks) == 0  # playable

    def test_original_boundary_conditions(self):
        """Test exact boundary between playable, discardable, and unknown."""
        for level in range(5):
            fireworks = jnp.zeros((5, 5))
            if level > 0:
                fireworks = fireworks.at[0, :level].set(1.0)
            
            for rank in range(5):
                card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
                status = aux.classify_card_status_original(card, fireworks)
                
                if rank < level:
                    assert status == 1, f"Rank {rank} < level {level} should be discardable"
                elif rank == level:
                    assert status == 0, f"Rank {rank} == level {level} should be playable"
                else:
                    assert status == 2, f"Rank {rank} > level {level} should be unknown"


class TestOriginalVsExtendedLogic:
    """Compare original and extended logic to understand differences."""

    def setup_method(self):
        self.fireworks = jnp.zeros((5, 5))
        self.deck = jnp.zeros((50, 5, 5))
        self.discard_pile = jnp.zeros((50, 5, 5))
        self.num_cards_of_rank = jnp.array([3, 2, 2, 2, 1])

    def test_playable_cards_match_both_logics(self):
        """Playable cards should be the same in both logics."""
        fireworks = self.fireworks.at[0, :2].set(1.0)  # Red at level 2
        card = jnp.zeros((5, 5)).at[0, 2].set(1.0)  # Red 3
        
        original_status = aux.classify_card_status_original(card, fireworks)
        extended_status = aux.classify_card_status(
            card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank
        )
        
        # Both should classify as playable (0)
        assert original_status == 0
        assert extended_status == 0

    def test_discardable_obsolete_cards_match(self):
        """Cards below firework level should be discardable in both logics."""
        fireworks = self.fireworks.at[0, :3].set(1.0)  # Red at level 3
        card = jnp.zeros((5, 5)).at[0, 1].set(1.0)  # Red 2
        
        original_status = aux.classify_card_status_original(card, fireworks)
        extended_status = aux.classify_card_status(
            card, fireworks, self.deck, self.discard_pile, self.num_cards_of_rank
        )
        
        # Both should classify as discardable (1)
        assert original_status == 1
        assert extended_status == 1

    def test_difference_impossible_prerequisites(self):
        """Extended logic marks cards as discardable when prerequisites impossible."""
        fireworks = self.fireworks.at[0, 0].set(1.0)  # Red 1 played
        
        # Discard all Red 2s (rank 1)
        discard_pile = self.discard_pile.at[0, 0, 1].set(1.0)
        discard_pile = discard_pile.at[1, 0, 1].set(1.0)
        
        card = jnp.zeros((5, 5)).at[0, 3].set(1.0)  # Red 4
        
        # Original logic: rank 3 > level 1 → unknown
        original_status = aux.classify_card_status_original(card, fireworks)
        assert original_status == 2  # unknown
        
        # Extended logic: Red 2 impossible → discardable
        extended_status = aux.classify_card_status(
            card, fireworks, self.deck, discard_pile, self.num_cards_of_rank
        )
        assert extended_status == 1  # discardable

    def test_original_simpler_no_prereq_check(self):
        """Original logic doesn't check if prerequisites are impossible."""
        # Even if all rank 1 cards are discarded, original logic still marks
        # rank 2+ as unknown (not discardable)
        
        # Discard all Red 1s
        discard_pile = self.discard_pile.at[0, 0, 0].set(1.0)
        discard_pile = discard_pile.at[1, 0, 0].set(1.0)
        discard_pile = discard_pile.at[2, 0, 0].set(1.0)
        
        card = jnp.zeros((5, 5)).at[0, 1].set(1.0)  # Red 2
        
        # Original: rank 1 > level 0 → unknown (doesn't care about prerequisites)
        original_status = aux.classify_card_status_original(card, self.fireworks)
        assert original_status == 2  # unknown
        
        # Extended: Red 1 impossible → discardable
        extended_status = aux.classify_card_status(
            card, self.fireworks, self.deck, discard_pile, self.num_cards_of_rank
        )
        assert extended_status == 1  # discardable

    def test_both_logics_agree_on_empty_cards(self):
        """Empty cards should be unknown in both logics."""
        card = jnp.zeros((5, 5))
        
        original_status = aux.classify_card_status_original(card, self.fireworks)
        extended_status = aux.classify_card_status(
            card, self.fireworks, self.deck, self.discard_pile, self.num_cards_of_rank
        )
        
        assert original_status == 2  # unknown
        assert extended_status == 2  # unknown


class TestExtractHandStatusOriginal:
    """Test extract_hand_status with discardable_logic='original'."""

    def setup_method(self):
        self.key = jax.random.PRNGKey(42)
        self.env = Hanabi(num_agents=2, num_colors=5, num_ranks=5, hand_size=5)

    def test_original_logic_initial_state(self):
        """Test extract_hand_status with original logic at game start."""
        key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)
        
        # Create batch
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        
        # Extract with original logic
        hand_status, hand_slot_mask = aux.extract_hand_status(
            env_states, discardable_logic="original"
        )
        
        # Check shapes
        assert hand_status.shape == (2, 15)  # 2 players, 5 cards * 3 statuses
        assert hand_slot_mask.shape == (2, 5)
        
        # All slots should be occupied
        assert jnp.all(hand_slot_mask == 1.0)
        
        # Reshape to check individual cards
        hand_status_reshaped = hand_status.reshape(2, 5, 3)
        
        # Check each player's hand
        for player_idx in range(2):
            player_hand = state.player_hands[player_idx]
            player_status = hand_status_reshaped[player_idx]
            
            for card_idx in range(5):
                card = player_hand[card_idx]
                color, rank = aux.get_card_color_and_rank(card)
                status = player_status[card_idx]
                
                # At start, rank 0 should be playable, others unknown
                if rank == 0:
                    assert jnp.allclose(status, jnp.array([1.0, 0.0, 0.0]))
                else:
                    assert jnp.allclose(status, jnp.array([0.0, 0.0, 1.0]))

    def test_original_logic_after_playing_cards(self):
        """Test original logic classification after playing some cards."""
        key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)
        
        # Play a Red 1 card
        player_hand = state.player_hands[jnp.argmax(state.cur_player_idx)]
        for card_idx in range(5):
            card = player_hand[card_idx]
            color, rank = aux.get_card_color_and_rank(card)
            if color == 0 and rank == 0:  # Red 1
                red_1_action = 5 + card_idx
                current_player = self.env.agents[jnp.argmax(state.cur_player_idx)]
                actions = {
                    agent: red_1_action if agent == current_player else 0 
                    for agent in self.env.agents
                }
                key, subkey = jax.random.split(key)
                obs, state, rewards, dones, info = self.env.step_env(subkey, state, actions)
                break
        
        # Create batch
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        
        # Extract with original logic
        hand_status, hand_slot_mask = aux.extract_hand_status(
            env_states, discardable_logic="original"
        )
        
        hand_status_reshaped = hand_status.reshape(2, 5, 3)
        
        # Check classification
        for player_idx in range(2):
            player_hand = state.player_hands[player_idx]
            player_status = hand_status_reshaped[player_idx]
            
            for card_idx in range(5):
                card = player_hand[card_idx]
                if card.any():
                    color, rank = aux.get_card_color_and_rank(card)
                    status = player_status[card_idx]
                    
                    if color == 0:  # Red cards
                        firework_level = int(state.fireworks[0].sum())
                        if rank < firework_level:
                            # Discardable
                            assert jnp.allclose(status, jnp.array([0.0, 1.0, 0.0]))
                        elif rank == firework_level:
                            # Playable
                            assert jnp.allclose(status, jnp.array([1.0, 0.0, 0.0]))
                        else:
                            # Unknown
                            assert jnp.allclose(status, jnp.array([0.0, 0.0, 1.0]))

    def test_original_vs_extended_on_same_state(self):
        """Compare original and extended logic on the same game state."""
        key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)
        
        # Play a few moves
        for _ in range(3):
            legal_moves = self.env.get_legal_moves(state)
            current_player = self.env.agents[jnp.argmax(state.cur_player_idx)]
            key, subkey = jax.random.split(key)
            legal_actions = jnp.where(legal_moves[current_player] > 0)[0]
            if len(legal_actions) > 0:
                action = legal_actions[0]
                actions = {
                    agent: action if agent == current_player else 0 
                    for agent in self.env.agents
                }
                key, subkey = jax.random.split(key)
                obs, state, rewards, dones, info = self.env.step_env(subkey, state, actions)
        
        # Create batch
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        
        # Extract with both logics
        hand_status_original, mask_original = aux.extract_hand_status(
            env_states, discardable_logic="original"
        )
        hand_status_extended, mask_extended = aux.extract_hand_status(
            env_states, discardable_logic="extended"
        )
        
        # Masks should be identical
        assert jnp.allclose(mask_original, mask_extended)
        
        # Statuses might differ (extended can mark more cards as discardable)
        # But playable cards should match
        hand_status_original = hand_status_original.reshape(2, 5, 3)
        hand_status_extended = hand_status_extended.reshape(2, 5, 3)
        
        for player_idx in range(2):
            for card_idx in range(5):
                orig_status = hand_status_original[player_idx, card_idx]
                ext_status = hand_status_extended[player_idx, card_idx]
                
                # If original says playable, extended should too
                if orig_status[0] == 1.0:
                    assert ext_status[0] == 1.0, \
                        "Playable cards should match between logics"

    def test_original_logic_batched_envs(self):
        """Test original logic with multiple environments."""
        key = self.key
        num_envs = 4
        
        # Create multiple states
        env_states_list = []
        keys = jax.random.split(key, num_envs)
        
        for k in keys:
            _, s = self.env.reset(k)
            env_states_list.append(s)
        
        # Stack into batch
        env_states = jax.tree.map(lambda *args: jnp.stack(args), *env_states_list)
        
        # Extract with original logic
        hand_status, hand_slot_mask = aux.extract_hand_status(
            env_states, discardable_logic="original"
        )
        
        # Check shapes
        assert hand_status.shape == (num_envs * 2, 15)
        assert hand_slot_mask.shape == (num_envs * 2, 5)
        
        # Verify all statuses are valid one-hot
        hand_status_reshaped = hand_status.reshape(num_envs * 2, 5, 3)
        assert jnp.allclose(hand_status_reshaped.sum(axis=2), 1.0)

    def test_original_logic_jit_compatible(self):
        """Test that extract_hand_status with original logic is JIT-compatible."""
        key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)
        
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        
        # JIT compile
        jitted_extract = jax.jit(aux.extract_hand_status, static_argnums=(1,))
        
        # Run with original logic
        hand_status, hand_slot_mask = jitted_extract(env_states, "original")
        
        # Should work without errors
        assert hand_status.shape == (2, 15)
        assert hand_slot_mask.shape == (2, 5)


class TestOriginalLogicRealGameScenarios:
    """Test original logic with realistic Hanabi game scenarios."""

    def setup_method(self):
        self.key = jax.random.PRNGKey(123)
        self.env = Hanabi(num_agents=2, num_colors=5, num_ranks=5, hand_size=5)

    def test_original_progressive_firework_building(self):
        """Test classification as fireworks progress for one color."""
        fireworks = jnp.zeros((5, 5))
        
        # Test Red color progression
        for level in range(6):  # 0 to 5
            if level > 0:
                fireworks = fireworks.at[0, :level].set(1.0)
            
            for rank in range(5):
                card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
                status = aux.classify_card_status_original(card, fireworks)
                
                if rank < level:
                    assert status == 1, f"Rank {rank} should be discardable at level {level}"
                elif rank == level and level < 5:
                    assert status == 0, f"Rank {rank} should be playable at level {level}"
                else:
                    assert status == 2, f"Rank {rank} should be unknown at level {level}"

    def test_original_multi_color_game_state(self):
        """Test a complex multi-color game state."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :5].set(1.0)  # Red complete
        fireworks = fireworks.at[1, :3].set(1.0)  # Yellow at 3
        fireworks = fireworks.at[2, :1].set(1.0)  # Green at 1
        # Blue and White at 0
        
        test_cases = [
            # (color, rank, expected_status)
            (0, 0, 1),  # Red 1: discardable (complete)
            (0, 4, 1),  # Red 5: discardable (complete)
            (1, 2, 1),  # Yellow 3: discardable (< level 3)
            (1, 3, 0),  # Yellow 4: playable (== level 3)
            (1, 4, 2),  # Yellow 5: unknown (> level 3)
            (2, 0, 1),  # Green 1: discardable (< level 1)
            (2, 1, 0),  # Green 2: playable (== level 1)
            (2, 2, 2),  # Green 3: unknown (> level 1)
            (3, 0, 0),  # Blue 1: playable (== level 0)
            (3, 1, 2),  # Blue 2: unknown (> level 0)
            (4, 0, 0),  # White 1: playable (== level 0)
        ]
        
        for color, rank, expected in test_cases:
            card = jnp.zeros((5, 5)).at[color, rank].set(1.0)
            status = aux.classify_card_status_original(card, fireworks)
            assert status == expected, \
                f"Card ({color}, {rank}) should be {expected}, got {status}"

    def test_original_symmetric_behavior(self):
        """Test that all colors behave identically at same firework level."""
        # Set all colors to level 2
        fireworks = jnp.zeros((5, 5))
        for color in range(5):
            fireworks = fireworks.at[color, :2].set(1.0)
        
        # Check that all colors have identical classification at each rank
        for rank in range(5):
            statuses = []
            for color in range(5):
                card = jnp.zeros((5, 5)).at[color, rank].set(1.0)
                status = aux.classify_card_status_original(card, fireworks)
                statuses.append(int(status))  # Convert to Python int for set comparison
            
            # All should be the same
            assert len(set(statuses)) == 1, \
                f"Rank {rank} should have same status across all colors"

    def test_original_end_game_scenario(self):
        """Test classification in end-game with most colors complete."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :].set(1.0)   # Red complete
        fireworks = fireworks.at[1, :].set(1.0)   # Yellow complete
        fireworks = fireworks.at[2, :].set(1.0)   # Green complete
        fireworks = fireworks.at[3, :].set(1.0)   # White complete
        fireworks = fireworks.at[4, :4].set(1.0)  # Blue at 4
        
        # Blue 5 should be playable
        card_b5 = jnp.zeros((5, 5)).at[4, 4].set(1.0)
        assert aux.classify_card_status_original(card_b5, fireworks) == 0
        
        # All other color cards should be discardable
        for color in range(4):
            for rank in range(5):
                card = jnp.zeros((5, 5)).at[color, rank].set(1.0)
                assert aux.classify_card_status_original(card, fireworks) == 1

    def test_original_consistency_with_actual_game(self):
        """Test that original logic produces valid results in actual game."""
        key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)
        
        # Play 10 random moves
        for _ in range(10):
            legal_moves = self.env.get_legal_moves(state)
            current_player = self.env.agents[jnp.argmax(state.cur_player_idx)]
            
            key, subkey = jax.random.split(key)
            legal_actions = jnp.where(legal_moves[current_player] > 0)[0]
            if len(legal_actions) == 0:
                break
            
            action = jax.random.choice(subkey, legal_actions)
            actions = {
                agent: action if agent == current_player else 0 
                for agent in self.env.agents
            }
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = self.env.step_env(subkey, state, actions)
            
            if dones["__all__"]:
                break
        
        # Extract hand status
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        hand_status, hand_slot_mask = aux.extract_hand_status(
            env_states, discardable_logic="original"
        )
        
        # Verify consistency: manually check a few cards
        hand_status_reshaped = hand_status.reshape(2, 5, 3)
        
        for player_idx in range(2):
            player_hand = state.player_hands[player_idx]
            player_status = hand_status_reshaped[player_idx]
            
            for card_idx in range(5):
                card = player_hand[card_idx]
                if card.any():
                    color, rank = aux.get_card_color_and_rank(card)
                    firework_level = int(state.fireworks[color].sum())
                    
                    status = player_status[card_idx]
                    
                    # Verify classification matches simple rule
                    if rank < firework_level:
                        assert jnp.allclose(status, jnp.array([0.0, 1.0, 0.0])), \
                            "Should be discardable"
                    elif rank == firework_level:
                        assert jnp.allclose(status, jnp.array([1.0, 0.0, 0.0])), \
                            "Should be playable"
                    else:
                        assert jnp.allclose(status, jnp.array([0.0, 0.0, 1.0])), \
                            "Should be unknown"


class TestOriginalLogicTrickyEdgeCases:
    """Tricky edge cases for the original logic to ensure it handles all scenarios correctly."""

    def setup_method(self):
        self.fireworks = jnp.zeros((5, 5))

    def test_original_all_colors_different_progress(self):
        """Test with all 5 colors at different progress levels."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :].set(1.0)   # Red complete (5/5)
        fireworks = fireworks.at[1, :4].set(1.0)  # Yellow at 4/5
        fireworks = fireworks.at[2, :2].set(1.0)  # Green at 2/5
        fireworks = fireworks.at[3, 0].set(1.0)   # White at 1/5
        # Blue at 0/5
        
        # Red 1 - discardable (rank 0 < level 5)
        assert aux.classify_card_status_original(
            jnp.zeros((5, 5)).at[0, 0].set(1.0), fireworks
        ) == 1
        
        # Yellow 5 - playable (rank 4 == level 4)
        assert aux.classify_card_status_original(
            jnp.zeros((5, 5)).at[1, 4].set(1.0), fireworks
        ) == 0
        
        # Green 3 - playable (rank 2 == level 2)
        assert aux.classify_card_status_original(
            jnp.zeros((5, 5)).at[2, 2].set(1.0), fireworks
        ) == 0
        
        # White 2 - playable (rank 1 == level 1)
        assert aux.classify_card_status_original(
            jnp.zeros((5, 5)).at[3, 1].set(1.0), fireworks
        ) == 0
        
        # Blue 1 - playable (rank 0 == level 0)
        assert aux.classify_card_status_original(
            jnp.zeros((5, 5)).at[4, 0].set(1.0), fireworks
        ) == 0

    def test_original_all_rank_1_discarded_doesnt_matter(self):
        """Original logic doesn't care if all rank 1s are gone - higher ranks still unknown."""
        # The key difference: original logic doesn't check prerequisites
        # Even if all Red 1s are gone, Red 2+ are still "unknown" not "discardable"
        
        fireworks = jnp.zeros((5, 5))
        
        # All Red cards (2,3,4,5) should be unknown since firework level is 0
        # (Original logic doesn't check if they're achievable)
        for rank in range(1, 5):
            card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
            status = aux.classify_card_status_original(card, fireworks)
            assert status == 2, f"Red {rank+1} should be unknown (rank {rank} > level 0)"

    def test_original_last_copy_of_critical_card(self):
        """Test behavior with the last copy of rank 5 card."""
        card = jnp.zeros((5, 5)).at[0, 4].set(1.0)  # Red 5
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :4].set(1.0)  # Red 1-4 played
        
        # Red 5 should be playable (rank 4 == level 4)
        assert aux.classify_card_status_original(card, fireworks) == 0

    def test_original_card_exactly_matches_firework_level(self):
        """Test edge case where card rank equals firework level - already played scenario."""
        card = jnp.zeros((5, 5)).at[2, 2].set(1.0)  # Green 3 (rank 2)
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[2, 0].set(1.0)  # Green 1
        fireworks = fireworks.at[2, 1].set(1.0)  # Green 2
        fireworks = fireworks.at[2, 2].set(1.0)  # Green 3 - same as our card!
        
        # Card is discardable (rank 2 < level 3)
        assert aux.classify_card_status_original(card, fireworks) == 1

    def test_original_complete_color_all_cards_discardable(self):
        """Test that when a color is complete, all cards of that color are discardable."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[4, :].set(1.0)  # Blue complete
        
        # All Blue cards should be discardable (rank < level 5)
        for rank in range(5):
            card = jnp.zeros((5, 5)).at[4, rank].set(1.0)
            status = aux.classify_card_status_original(card, fireworks)
            assert status == 1, f"Blue {rank+1} should be discardable when Blue is complete"

    def test_original_zero_fireworks_all_rank_1_playable(self):
        """Test that with empty fireworks, all rank 1 cards are playable across all colors."""
        fireworks = jnp.zeros((5, 5))
        
        # All rank 0 (value 1) cards should be playable
        for color in range(5):
            card = jnp.zeros((5, 5)).at[color, 0].set(1.0)
            status = aux.classify_card_status_original(card, fireworks)
            assert status == 0, f"Rank 1 for color {color} should be playable"

    def test_original_near_end_game_scenario(self):
        """Test classification near end game with most cards played."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :].set(1.0)    # Red complete
        fireworks = fireworks.at[1, :].set(1.0)    # Yellow complete
        fireworks = fireworks.at[2, :].set(1.0)    # Green complete
        fireworks = fireworks.at[3, :].set(1.0)    # White complete
        fireworks = fireworks.at[4, :3].set(1.0)   # Blue at 3
        
        # Blue 4 should be playable (rank 3 == level 3)
        card_blue_4 = jnp.zeros((5, 5)).at[4, 3].set(1.0)
        assert aux.classify_card_status_original(card_blue_4, fireworks) == 0
        
        # Any card from complete colors should be discardable
        card_red_1 = jnp.zeros((5, 5)).at[0, 0].set(1.0)
        assert aux.classify_card_status_original(card_red_1, fireworks) == 1

    def test_original_second_to_last_rank(self):
        """Test rank 4 card when rank 5 is not yet played."""
        card = jnp.zeros((5, 5)).at[3, 3].set(1.0)  # White 4
        
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[3, :3].set(1.0)  # White 1,2,3 played
        
        # White 4 should be playable (rank 3 == level 3)
        assert aux.classify_card_status_original(card, fireworks) == 0

    def test_original_batch_classification_all_ranks_same_color(self):
        """Test classifying all 5 ranks of the same color simultaneously."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[2, 0].set(1.0)  # Green 1 played
        fireworks = fireworks.at[2, 1].set(1.0)  # Green 2 played
        
        # Create all 5 Green cards
        cards = jnp.stack([jnp.zeros((5, 5)).at[2, r].set(1.0) for r in range(5)])
        
        # Classify all at once using vmap
        statuses = jax.vmap(
            lambda c: aux.classify_card_status_original(c, fireworks)
        )(cards)
        
        # Green 1: discardable (rank 0 < level 2)
        assert statuses[0] == 1
        # Green 2: discardable (rank 1 < level 2)
        assert statuses[1] == 1
        # Green 3: playable (rank 2 == level 2)
        assert statuses[2] == 0
        # Green 4: unknown (rank 3 > level 2)
        assert statuses[3] == 2
        # Green 5: unknown (rank 4 > level 2)
        assert statuses[4] == 2

    def test_original_cross_color_independence(self):
        """Test that classification in one color doesn't affect other colors."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :].set(1.0)  # Red complete
        
        # Yellow cards should be unaffected - Yellow 1 should still be playable
        card_y1 = jnp.zeros((5, 5)).at[1, 0].set(1.0)
        assert aux.classify_card_status_original(card_y1, fireworks) == 0
        
        # Yellow 2 should be unknown (rank 1 > level 0)
        card_y2 = jnp.zeros((5, 5)).at[1, 1].set(1.0)
        assert aux.classify_card_status_original(card_y2, fireworks) == 2

    def test_original_empty_card_handling(self):
        """Test that empty cards are handled correctly."""
        empty_card = jnp.zeros((5, 5))
        
        # Empty cards should be classified as unknown (2)
        status = aux.classify_card_status_original(empty_card, self.fireworks)
        assert status == 2

    def test_original_all_rank_1_cards_at_start(self):
        """Test that all rank 1 cards are playable at the start of the game."""
        for color in range(5):
            card = jnp.zeros((5, 5)).at[color, 0].set(1.0)
            status = aux.classify_card_status_original(card, self.fireworks)
            assert status == 0, f"Rank 1 for color {color} should be playable"

    def test_original_rank_5_edge_case(self):
        """Test the edge case of rank 5 cards."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :4].set(1.0)  # Red 1-4 played
        
        # Red 5 should be playable (rank 4 == level 4)
        card_r5 = jnp.zeros((5, 5)).at[0, 4].set(1.0)
        assert aux.classify_card_status_original(card_r5, fireworks) == 0


class TestOriginalLogicAdvancedScenarios:
    """Advanced scenarios to test original logic comprehensively."""

    def setup_method(self):
        self.fireworks = jnp.zeros((5, 5))

    def test_original_transitive_ignorance(self):
        """Original logic doesn't propagate impossibility through chains."""
        # Unlike extended logic, original doesn't care if prerequisites are impossible
        # Red 2,3,4,5 will be "unknown" even if all Red 1s are discarded
        
        fireworks = jnp.zeros((5, 5))
        
        # All higher ranks are just "unknown" (rank > level 0)
        for rank in range(1, 5):
            card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
            status = aux.classify_card_status_original(card, fireworks)
            assert status == 2, f"Red {rank+1} should be unknown (original logic ignores prerequisites)"

    def test_original_obsolete_vs_unknown_boundary(self):
        """Test the exact boundary between obsolete and unknown."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :2].set(1.0)  # Red at level 2
        
        # Red 1 is obsolete (rank 0 < level 2)
        card_r1 = jnp.zeros((5, 5)).at[0, 0].set(1.0)
        assert aux.classify_card_status_original(card_r1, fireworks) == 1
        
        # Red 2 is also obsolete (rank 1 < level 2)
        card_r2 = jnp.zeros((5, 5)).at[0, 1].set(1.0)
        assert aux.classify_card_status_original(card_r2, fireworks) == 1
        
        # Red 3 is playable (rank 2 == level 2)
        card_r3 = jnp.zeros((5, 5)).at[0, 2].set(1.0)
        assert aux.classify_card_status_original(card_r3, fireworks) == 0
        
        # Red 4+ are unknown (rank > level 2)
        card_r4 = jnp.zeros((5, 5)).at[0, 3].set(1.0)
        assert aux.classify_card_status_original(card_r4, fireworks) == 2

    def test_original_boundary_sweep_all_levels(self):
        """Sweep through all firework levels and verify boundaries."""
        for level in range(6):  # 0 through 5
            fireworks = jnp.zeros((5, 5))
            if level > 0:
                fireworks = fireworks.at[0, :level].set(1.0)
            
            for rank in range(5):
                card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
                status = aux.classify_card_status_original(card, fireworks)
                
                if rank < level:
                    assert status == 1, f"Rank {rank} < level {level} should be discardable"
                elif rank == level and level < 5:
                    assert status == 0, f"Rank {rank} == level {level} should be playable"
                else:
                    assert status == 2, f"Rank {rank} > level {level} should be unknown"

    def test_original_complex_multi_color_state(self):
        """Test complex state with all colors at different levels."""
        fireworks = jnp.zeros((5, 5))
        levels = [5, 4, 3, 2, 1]  # Different level for each color
        
        for color, level in enumerate(levels):
            if level > 0:
                fireworks = fireworks.at[color, :level].set(1.0)
        
        # Test a variety of cards
        test_cases = [
            (0, 0, 1),  # Red 1 at level 5: discardable
            (0, 4, 1),  # Red 5 at level 5: discardable
            (1, 3, 1),  # Yellow 4 at level 4: discardable
            (1, 4, 0),  # Yellow 5 at level 4: playable
            (2, 2, 1),  # Green 3 at level 3: discardable
            (2, 3, 0),  # Green 4 at level 3: playable
            (2, 4, 2),  # Green 5 at level 3: unknown
            (3, 1, 1),  # White 2 at level 2: discardable
            (3, 2, 0),  # White 3 at level 2: playable
            (3, 3, 2),  # White 4 at level 2: unknown
            (4, 0, 1),  # Blue 1 at level 1: discardable
            (4, 1, 0),  # Blue 2 at level 1: playable
            (4, 2, 2),  # Blue 3 at level 1: unknown
        ]
        
        for color, rank, expected in test_cases:
            card = jnp.zeros((5, 5)).at[color, rank].set(1.0)
            status = aux.classify_card_status_original(card, fireworks)
            assert status == expected, \
                f"Card ({color}, {rank}) at level {levels[color]} should be {expected}, got {status}"

    def test_original_vmap_consistency(self):
        """Test that vmap produces consistent results."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :2].set(1.0)
        
        # Create batch of Red cards
        cards = jnp.stack([jnp.zeros((5, 5)).at[0, r].set(1.0) for r in range(5)])
        
        # Classify with vmap
        statuses_vmap = jax.vmap(
            lambda c: aux.classify_card_status_original(c, fireworks)
        )(cards)
        
        # Classify individually
        statuses_individual = jnp.array([
            aux.classify_card_status_original(cards[i], fireworks)
            for i in range(5)
        ])
        
        # Should match
        assert jnp.allclose(statuses_vmap, statuses_individual)

    def test_original_jit_stability(self):
        """Test that JIT compilation produces stable results."""
        fireworks = jnp.zeros((5, 5))
        fireworks = fireworks.at[0, :3].set(1.0)
        card = jnp.zeros((5, 5)).at[0, 2].set(1.0)
        
        # JIT compile
        jitted_classify = jax.jit(aux.classify_card_status_original)
        
        # Run multiple times
        results = [jitted_classify(card, fireworks) for _ in range(5)]
        
        # All should be identical
        assert all(results[0] == r for r in results)

    def test_original_symmetry_across_all_colors(self):
        """Test that behavior is symmetric across all colors."""
        for level in range(5):
            fireworks = jnp.zeros((5, 5))
            if level > 0:
                for color in range(5):
                    fireworks = fireworks.at[color, :level].set(1.0)
            
            # For each rank, all colors should give same status
            for rank in range(5):
                expected_status = None
                for color in range(5):
                    card = jnp.zeros((5, 5)).at[color, rank].set(1.0)
                    status = int(aux.classify_card_status_original(card, fireworks))
                    
                    if expected_status is None:
                        expected_status = status
                    else:
                        assert status == expected_status, \
                            f"At level {level}, rank {rank} should be same across colors"

    def test_original_progressive_levels_single_color(self):
        """Test progressive advancement through all levels for a single color."""
        for target_level in range(6):
            fireworks = jnp.zeros((5, 5))
            if target_level > 0:
                fireworks = fireworks.at[0, :target_level].set(1.0)
            
            for rank in range(5):
                card = jnp.zeros((5, 5)).at[0, rank].set(1.0)
                status = aux.classify_card_status_original(card, fireworks)
                
                # Verify correct classification
                if rank < target_level:
                    assert status == 1  # discardable
                elif rank == target_level and target_level < 5:
                    assert status == 0  # playable
                else:
                    assert status == 2  # unknown


class TestOriginalLogicPropertyTests:
    """Property-based tests for original logic."""

    def setup_method(self):
        self.num_colors = 5
        self.num_ranks = 5
        self.rng = np.random.default_rng(42)

    def test_original_playable_and_discardable_mutually_exclusive(self):
        """No card should be both playable (0) and discardable (1)."""
        for _ in range(20):
            # Random firework state
            fireworks = jnp.zeros((self.num_colors, self.num_ranks))
            levels = self.rng.integers(0, self.num_ranks + 1, size=self.num_colors)
            for c in range(self.num_colors):
                if levels[c] > 0:
                    fireworks = fireworks.at[c, :levels[c]].set(1.0)
            
            for c in range(self.num_colors):
                for r in range(self.num_ranks):
                    card = jnp.zeros((self.num_colors, self.num_ranks)).at[c, r].set(1.0)
                    status = int(aux.classify_card_status_original(card, fireworks))
                    
                    # Status should be 0, 1, or 2
                    assert status in [0, 1, 2]

    def test_original_status_matches_simple_rule(self):
        """Verify that status always matches the simple rule."""
        for _ in range(20):
            fireworks = jnp.zeros((self.num_colors, self.num_ranks))
            levels = self.rng.integers(0, self.num_ranks + 1, size=self.num_colors)
            for c in range(self.num_colors):
                if levels[c] > 0:
                    fireworks = fireworks.at[c, :levels[c]].set(1.0)
            
            for c in range(self.num_colors):
                level = int(levels[c])
                for r in range(self.num_ranks):
                    card = jnp.zeros((self.num_colors, self.num_ranks)).at[c, r].set(1.0)
                    status = int(aux.classify_card_status_original(card, fireworks))
                    
                    # Verify simple rule
                    if r < level:
                        assert status == 1, f"rank {r} < level {level} should be discardable"
                    elif r == level and level < 5:
                        assert status == 0, f"rank {r} == level {level} should be playable"
                    else:
                        assert status == 2, f"rank {r} > level {level} should be unknown"

    def test_original_deterministic_results(self):
        """Results should be deterministic for same inputs."""
        fireworks = jnp.zeros((self.num_colors, self.num_ranks))
        fireworks = fireworks.at[0, :2].set(1.0)
        fireworks = fireworks.at[1, :3].set(1.0)
        
        card = jnp.zeros((self.num_colors, self.num_ranks)).at[0, 1].set(1.0)
        
        # Call multiple times
        results = [aux.classify_card_status_original(card, fireworks) for _ in range(10)]
        
        # All should be identical
        assert all(results[0] == r for r in results)


def run_tests():
    """Simple test runner without pytest."""
    test_classes = [
        TestCardColorAndRank,
        TestCardPlayable,
        TestCardDiscardable,
        TestClassifyCardStatus,
        TestTrickyEdgeCases,
        TestHyperTrickyEdgeCases,
        TestRealHanabiIntegration,
        TestExtractHandStatus,
        TestAdvancedLogic,
        TestJAXCompatibility,
        TestGlobalProperties,
        TestOriginalLogic,
        TestOriginalVsExtendedLogic,
        TestExtractHandStatusOriginal,
        TestOriginalLogicRealGameScenarios,
        TestOriginalLogicTrickyEdgeCases,
        TestOriginalLogicAdvancedScenarios,
        TestOriginalLogicPropertyTests,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    print("=" * 70)
    print("Running Tests")
    print("=" * 70)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 70)
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            test_instance = test_class()
            
            # Run setup if it exists
            if hasattr(test_instance, 'setup_method'):
                test_instance.setup_method()
            
            try:
                # Run the test method
                getattr(test_instance, method_name)()
                passed_tests += 1
                print(f"  ✓ {method_name}")
            except AssertionError as e:
                failed_tests += 1
                print(f"  ✗ {method_name}")
                print(f"    AssertionError: {e}")
            except Exception as e:
                failed_tests += 1
                print(f"  ✗ {method_name}")
                print(f"    Error: {e}")
                traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed_tests}/{total_tests} passed, {failed_tests} failed")
    print("=" * 70)
    
    return failed_tests == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)


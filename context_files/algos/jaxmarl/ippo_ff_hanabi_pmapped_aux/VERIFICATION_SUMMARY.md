# Verification of Card Encoding/Decoding Logic in aux.py

## Summary
✅ **VERIFIED**: The card encoding/decoding logic in `is_card_discardable` is consistent with the Hanabi environment implementation.

## Card Encoding (Hanabi Environment)

### Card Structure
- **Shape**: `(num_colors, num_ranks)` where typically `num_colors=5`, `num_ranks=5`
- **Encoding**: One-hot encoded, e.g., card[color, rank] = 1 for that specific card
- **Empty cards**: All zeros

**Source**: `hanabi_game.py` lines 458-459:
```python
card = jnp.zeros((self.num_colors, self.num_ranks))
card = card.at[color, rank].set(1)
```

## Card Decoding

### In Hanabi Environment (`hanabi.py` lines 601-602)
```python
color = jnp.argmax(card.sum(axis=1), axis=0)  # sum over ranks, argmax over colors
rank = jnp.argmax(card.sum(axis=0), axis=0)   # sum over colors, argmax over ranks
```

### In aux.py (`aux.py` lines 18-19)
```python
color = jnp.argmax(card.sum(axis=1))  # sum over ranks, argmax over colors
rank = jnp.argmax(card.sum(axis=0))   # sum over colors, argmax over ranks
```

✅ **Match**: Both use the same decoding logic.

## Fireworks Structure

### Encoding
- **Shape**: `(num_colors, num_ranks)`
- **Meaning**: `fireworks[color, rank] = 1` if that rank has been completed for that color
- **Incremental**: e.g., `[1,1,1,0,0]` means ranks 0, 1, 2 are complete for that color

**Source**: `hanabi_game.py` lines 217-241:
- Line 219: `is_valid_play = rank == jnp.sum(color_fireworks)` - A card is playable if its rank equals the count of completed ranks
- Lines 238-240: Updates fireworks by setting `fireworks[color, next_rank] = 1`

### Current Firework Level Calculation

**In aux.py** (line 65):
```python
current_firework_level = jnp.sum(fireworks[color])
```

✅ **Correct**: Summing `fireworks[color]` gives the count of completed ranks for that color, which is the next playable rank.

**Example**: If `fireworks[Red] = [1,1,0,0,0]`, then `sum = 2`, meaning ranks 0 and 1 are done, and rank 2 is next.

## Playable Check (`is_card_playable`)

**aux.py** line 45:
```python
is_playable = (rank == current_firework_level) & (~is_empty)
```

✅ **Correct**: Matches the Hanabi environment logic at `hanabi_game.py` line 219.

## Discard Pile Structure

### Encoding
- **Shape**: `(deck_size, num_colors, num_ranks)` where `deck_size = sum(num_cards_of_rank) * num_colors`
- **Meaning**: Each position along the first dimension holds a one-hot encoded card (or all zeros if unused)

**Source**: 
- `hanabi_game.py` line 108: `discard_pile = jnp.zeros_like(deck)`
- Lines 246-248: `discard_pile.at[state.num_cards_discarded].set(discarded_card)`

### Counting Discarded Cards

**In aux.py** (line 87):
```python
discarded_count = jnp.sum(discard_pile[:, color, prereq_rank])
```

✅ **Correct**: This sums over all positions in the discard pile for a specific (color, rank) combination, counting how many times that card has been discarded.

## Card Distribution (`num_cards_of_rank`)

**Standard Hanabi**: `[3, 2, 2, 2, 1]`
- Rank 0: 3 copies per color (15 total)
- Rank 1: 2 copies per color (10 total)
- Rank 2: 2 copies per color (10 total)
- Rank 3: 2 copies per color (10 total)
- Rank 4: 1 copy per color (5 total)

**In Hanabi Environment**: `hanabi.py` line 26, `hanabi_game.py` line 49
**In aux.py**: line 172

✅ **Match**: Both use `[3, 2, 2, 2, 1]`

### Interpretation
`num_cards_of_rank[rank]` gives the total number of cards of that rank **per color**.

## Discardable Logic (`is_card_discardable`)

### Condition 1: Already Obsolete
```python
condition1 = (rank < current_firework_level) & (~is_empty)
```

✅ **Correct**: If a card's rank is less than the current firework level, it's already been played and is obsolete.

### Condition 2: Prerequisite Impossible
For each prerequisite rank between `current_firework_level` and the card's `rank`:
1. Count played cards: `fireworks[color, prereq_rank]` (0 or 1)
2. Count discarded cards: `jnp.sum(discard_pile[:, color, prereq_rank])`
3. Check if all copies are gone: `played_count + discarded_count >= num_cards_of_rank[prereq_rank]`

✅ **Correct Logic**: 
- If all copies of a prerequisite rank are played/discarded, that rank is impossible to obtain
- If any prerequisite is impossible, higher ranks can never be played
- Only checks ranks in range `[current_firework_level, rank)` which are actual prerequisites

### Example Scenario
- Color: Red (0)
- Card rank: 3
- Current firework level: 1 (rank 0 completed)
- `num_cards_of_rank = [3, 2, 2, 2, 1]`

**Check rank 1 (prerequisite)**:
- `played_count = fireworks[Red, 1]` (0 or 1)
- `discarded_count = sum(discard_pile[:, Red, 1])`
- `total_cards = num_cards_of_rank[1] = 2`
- If `played_count + discarded_count >= 2`, rank 1 is impossible
- Since rank 3 requires rank 1 to be played first (through rank 2), the Red 3 is discardable

**Check rank 2 (prerequisite)**:
- Same logic applies
- If rank 2 is impossible, Red 3 is discardable

✅ **Correct**: This correctly identifies cards that can never be played due to missing prerequisites.

## Edge Cases

### Empty Cards
**aux.py** handles empty cards by:
1. Detecting: `is_empty = ~card.any()` (line 17)
2. Setting color/rank to -1 for empty cards (lines 22-23)
3. Using `jnp.where(color >= 0, ...)` to avoid using invalid indices (lines 38-42, 63-67, 78-82, 86-89)
4. Marking empty cards as "unknown" status (line 151)

✅ **Safe**: While JAX evaluates both branches of `jnp.where`, the final result correctly handles empty cards by returning status=2 (unknown).

## Potential Issues

### 1. Negative Indexing with color=-1
**Location**: Lines 40, 65, 80, 87

When a card is empty, `color = -1`. In JAX, `fireworks[-1]` or `discard_pile[:, -1, ...]` will index the last element (wrap-around). However, the `jnp.where(color >= 0, ...)` guards ensure the result from negative indexing is not used.

**Status**: ⚠️ **Minor concern** - While functionally safe, it's not ideal. Consider adding a comment or restructuring to make it clearer that negative indices are intentionally ignored.

### 2. Fixed max_rank=5
**Location**: Line 99

The code hardcodes `max_rank = 5`, which assumes standard Hanabi. This is consistent with the environment default but could be made more flexible.

**Status**: ✅ **Acceptable** - Matches the environment's standard configuration.

## Conclusion

✅ **VERIFIED**: The card encoding/decoding logic in `is_card_discardable` is **correct and consistent** with the Hanabi environment implementation.

### Key Findings:
1. ✅ Card encoding/decoding matches the Hanabi environment
2. ✅ Fireworks structure interpretation is correct
3. ✅ Discard pile indexing is correct
4. ✅ `num_cards_of_rank` interpretation is correct
5. ✅ Playable logic is correct
6. ✅ Discardable logic is correct and handles both conditions properly
7. ✅ Empty cards are handled safely

### Minor Recommendations:
1. Add comments explaining that negative indices from empty cards are intentionally guarded by `jnp.where`
2. Consider extracting `num_cards_of_rank` from the environment config if it might vary


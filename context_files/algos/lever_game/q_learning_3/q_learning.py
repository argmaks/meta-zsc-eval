"""
Independent Q-learning for the TwoPlayerLeverGame.

This script trains two separate Q-tables – one for each player – with an
ε-greedy exploration strategy that decays over time.  Because the underlying
environment terminates after a single step, the state–space is trivial
(step_count == 0 before the move).  Still, we keep the code generic so it can
be extended to multi-step environments with minimal changes.

Run from the project root:

    python q_learning.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np

from lever_game import TwoPlayerLeverGame


@dataclass
class HyperParams:
    episodes: int = 500            # training episodes
    alpha: float = 0.1                # learning rate
    gamma: float = 0.95               # discount factor
    epsilon_start: float = 1.0        # initial exploration rate
    epsilon_min: float = 0.05         # minimum exploration
    epsilon_decay: float = 0.9995     # multiplicative decay after each episode
    seed: int = 42                    # RNG seed for reproducibility


class IndependentQLearner:
    """
    Learn two independent Q-tables – one per player – in a fully cooperative,
    one-shot common-payoff game.
    """

    def __init__(self, env: TwoPlayerLeverGame, hp: HyperParams) -> None:
        self.env = env
        self.hp = hp

        # The environment observation is trivial, but we keep a single-state Q-table.
        num_states = 1  # (step_count == 0)
        num_actions = env.num_levers

        # Q-tables: shape (num_states, num_actions)
        self.q_player1 = np.zeros((num_states, num_actions), dtype=np.float32)
        self.q_player2 = np.zeros_like(self.q_player1)

        # Local RNG to avoid global numpy state pollution
        self.rng = np.random.default_rng(hp.seed)

    # --------------------------------------------------------------------- #
    # POLICY                                                                
    # --------------------------------------------------------------------- #
    def _epsilon_greedy(self, q_row: np.ndarray, epsilon: float) -> int:
        """Return an action using ε-greedy exploration."""
        if self.rng.random() < epsilon:
            return self.rng.integers(0, q_row.shape[0])
        # If multiple max values, choose randomly among them
        max_q = np.max(q_row)
        best_actions = np.flatnonzero(np.isclose(q_row, max_q))
        return int(self.rng.choice(best_actions))

    # --------------------------------------------------------------------- #
    # TRAIN                                                                 
    # --------------------------------------------------------------------- #
    def train(self) -> None:
        """Main training loop."""
        epsilon = self.hp.epsilon_start

        for ep in range(1, self.hp.episodes + 1):
            state, _ = self.env.reset(seed=self.hp.seed + ep)
            state_idx = 0  # single state

            # Select actions for both agents
            a1 = self._epsilon_greedy(self.q_player1[state_idx], epsilon)
            a2 = self._epsilon_greedy(self.q_player2[state_idx], epsilon)

            (next_state, reward, terminated, _, _info) = self.env.step((a1, a2))
            next_state_idx = 0  # still single state

            # Synchronous Q-updates
            td_target = reward + self.hp.gamma * np.max(
                self.q_player1[next_state_idx]
            )
            td_error = td_target - self.q_player1[state_idx, a1]
            self.q_player1[state_idx, a1] += self.hp.alpha * td_error

            td_target2 = reward + self.hp.gamma * np.max(
                self.q_player2[next_state_idx]
            )
            td_error2 = td_target2 - self.q_player2[state_idx, a2]
            self.q_player2[state_idx, a2] += self.hp.alpha * td_error2

            # Decay ε
            epsilon = max(self.hp.epsilon_min, epsilon * self.hp.epsilon_decay)

        avg_reward = self.evaluate(100)
        print(f"[TRAINING COMPLETE] Seed: {self.hp.seed} | Episodes: {self.hp.episodes} | Avg Reward: {avg_reward:.3f}")

    # --------------------------------------------------------------------- #
    # EVALUATION                                                            
    # --------------------------------------------------------------------- #
    def greedy_policy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the greedy (deterministic) policies for both players."""
        return np.argmax(self.q_player1, axis=1), np.argmax(self.q_player2, axis=1)

    def evaluate(self, n_episodes: int = 1_000) -> float:
        """
        Play with greedy policies and return the mean reward per episode.
        """
        greedy_a1, greedy_a2 = self.greedy_policy()
        total_reward = 0.0

        for ep in range(n_episodes):
            self.env.reset(seed=self.hp.seed + 10_000 + ep)
            (obs, reward, *_rest) = self.env.step((int(greedy_a1[0]), int(greedy_a2[0])))
            total_reward += reward

        return float(total_reward / n_episodes)


    # --------------------------------------------------------------------- #
    # STATIC HELPERS                                                        
    # --------------------------------------------------------------------- #
    @staticmethod
    def default() -> "IndependentQLearner":
        """Return a learner with default hyper-parameters and environment."""
        hp = HyperParams()
        env = TwoPlayerLeverGame(random_seed=hp.seed)
        return IndependentQLearner(env, hp)


# ------------------------------------------------------------------------- #
# MULTI-SEED  TRAINING + CROSS-PLAY                                         #
# ------------------------------------------------------------------------- #
def train_multiple_seeds(n_seeds: int, episodes: int) -> list[IndependentQLearner]:
    """
    Train `n_seeds` independent pairs of agents, one random seed each.

    Returns
    -------
    list[IndependentQLearner]
        Trained learners in the order of their seeds (0…n_seeds-1).
    """
    learners: list[IndependentQLearner] = []
    for seed in range(n_seeds):
        hp = HyperParams(episodes=episodes, seed=seed)
        env = TwoPlayerLeverGame(random_seed=seed)
        learner = IndependentQLearner(env, hp)
        learner.train()
        learners.append(learner)
    return learners


def cross_play_matrix(learners: list[IndependentQLearner]) -> np.ndarray:
    """
    Compute the cross-play reward matrix.

    Element (i, j) is the expected reward (one step) when Player-1 uses the
    greedy policy learned with seed `i` and Player-2 uses the greedy policy
    learned with seed `j`.
    """
    n = len(learners)
    matrix = np.zeros((n, n), dtype=np.float32)

    # All envs share the same reward vector.
    reward_vec = learners[0].env.rewards

    greedy = [l.greedy_policy() for l in learners]  # list[(a1_arr, a2_arr)]
    for i in range(n):
        a1 = int(greedy[i][0][0])  # player-1 action from seed i
        for j in range(n):
            a2 = int(greedy[j][1][0])  # player-2 action from seed j
            matrix[i, j] = reward_vec[a1] if a1 == a2 else 0.0
    return matrix


# ------------------------------------------------------------------------- #
# CLI ENTRY POINT                                                           #
# ------------------------------------------------------------------------- #
def main() -> None:
    """
    Hard-coded entry point (no CLI parsing).

    Trains `n_seeds` pairs of agents, ,
    prints the cross-play matrix and key statistics.
    """
    n_seeds = 5
    episodes = 5000
    
    print("[Q-LEARNING EXPERIMENT] Independent learning on TwoPlayerLeverGame")
    print(f"Goal: Train {n_seeds} agent pairs, evaluate cooperation via cross-play")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # Train                                                               #
    # ------------------------------------------------------------------ #
    print(f"[EXPERIMENT START] Training {n_seeds} independent Q-learning pairs | Episodes per pair: {episodes}")
    print("-" * 80)
    learners = train_multiple_seeds(n_seeds, episodes)
    print("-" * 80)

    # ------------------------------------------------------------------ #
    # Persistence & metrics                                              #
    # ------------------------------------------------------------------ #
    script_dir = pathlib.Path(__file__).parent
    trained_agents_root = script_dir / "trained_agents"
    trained_agents_root.mkdir(parents=True, exist_ok=True)

    # Save Q-tables and gather self-play rewards
    self_play_rewards = []
    for idx, lrn in enumerate(learners):
        seed_dir = trained_agents_root / f"seed_{idx}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        with open(seed_dir / "lever_game_q_learning_agent1.pkl", "wb") as f1:
            pickle.dump(lrn.q_player1, f1)
        with open(seed_dir / "lever_game_q_learning_agent2.pkl", "wb") as f2:
            pickle.dump(lrn.q_player2, f2)

        self_play_rewards.append(lrn.evaluate())

    # Cross-play matrix & statistics
    matrix = cross_play_matrix(learners).astype(float)
    avg_self_play_reward = float(np.mean(self_play_rewards))
    avg_cross_play_reward = float(np.mean(matrix[np.triu_indices(n_seeds, k=1)]))

    results = {
        "cross_play_matrix": matrix.tolist(),
        "avg_self_play_reward": avg_self_play_reward,
        "avg_cross_play_reward": avg_cross_play_reward,
    }
    with open(script_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # ------------------------------------------------------------------ #
    # Output                                                             #
    # ------------------------------------------------------------------ #
    print()
    print("[CROSS-PLAY EVALUATION]")
    print(f"Matrix dimensions: {n_seeds}x{n_seeds} (Player1_seed × Player2_seed)")
    print("Cross-play reward matrix:")
    print(matrix)
    
    print()
    print("[PERFORMANCE SUMMARY]")
    print(f"• Self-play reward (diagonal avg):  {avg_self_play_reward:.3f}")
    print(f"• Cross-play reward (off-diag avg): {avg_cross_play_reward:.3f}")
    print(f"• Self-play cross-play gap: {avg_self_play_reward - avg_cross_play_reward:.3f}")




if __name__ == "__main__":
    main()

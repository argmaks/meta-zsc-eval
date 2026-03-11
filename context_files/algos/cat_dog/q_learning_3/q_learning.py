"""
Independent Q–learning for the Cat-Dog turn-based cooperative game.

We train two completely independent Q-tables – one for Alice and one for Bob – 
using ε-greedy exploration.  Each agent only learns from the steps when it is 
its turn; both receive the true cooperative reward returned by the environment.

"""

from __future__ import annotations

import random
import pickle
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import gymnasium as gym

from cat_dog import CatDogGame


@dataclass
class AgentConfig:
    lr: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.9995
    min_epsilon: float = 0.1
    action_space_size: int = 4  # maximum size across both agents


class IndependentQLearningAgent:
    """Tabular ε-greedy Q-learning agent."""

    def __init__(self, name: str, cfg: AgentConfig):
        self.name = name
        self.cfg = cfg
        # Q-table:  state(tuple of ints) -> np.ndarray[action_space_size]
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(cfg.action_space_size, dtype=np.float32)
        )

    # --------------------------------------------------------------------- #
    # Acting / learning
    # --------------------------------------------------------------------- #
    def select_action(self, state: Tuple[int, ...], valid_actions: List[int]) -> int:
        """ε-greedy over *valid* actions only."""
        if random.random() < self.cfg.epsilon:
            return random.choice(valid_actions)

        q_vals = self.q_table[state]

        # Mask invalid actions to -inf so they are never selected.
        masked = np.full_like(q_vals, -np.inf)
        masked[valid_actions] = q_vals[valid_actions]
        return int(np.argmax(masked))

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        terminated: bool,
    ) -> None:
        """Standard one-step Q-learning update."""
        best_next = 0.0 if terminated else np.max(self.q_table[next_state])
        td_target = reward + self.cfg.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.cfg.lr * td_error

    def decay_epsilon(self) -> None:
        self.cfg.epsilon = max(
            self.cfg.min_epsilon, self.cfg.epsilon * self.cfg.epsilon_decay
        )

    # --------------------------------------------------------------------- #
    # Serialisation helpers
    # --------------------------------------------------------------------- #
    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path: Path) -> None:
        with path.open("rb") as f:
            data: Dict[Tuple[int, ...], np.ndarray] = pickle.load(f)
            self.q_table.update(data)


# ------------------------------------------------------------------------- #
# Training & evaluation loops
# ------------------------------------------------------------------------- #
def train(episodes: int, seed: int | None = 0, verbose: bool = False) -> Tuple[IndependentQLearningAgent, IndependentQLearningAgent]:
    """Train Alice and Bob independently for a number of episodes."""
    env: gym.Env = CatDogGame()
    if seed is not None:
        env.reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)

    alice_cfg = AgentConfig()
    bob_cfg = AgentConfig()
    alice_agent = IndependentQLearningAgent("Alice", alice_cfg)
    bob_agent = IndependentQLearningAgent("Bob", bob_cfg)

    reward_window: List[float] = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        terminated = False
        total_reward = 0.0
        alice_transitions: List[Tuple[Tuple[int, ...], int, Tuple[int, ...], bool]] = []
        bob_transitions: List[Tuple[Tuple[int, ...], int, Tuple[int, ...], bool]] = []

        while not terminated:
            state = tuple(int(x) for x in obs)

            agent = alice_agent if env.turn == 0 else bob_agent
            valid_actions = info["valid_actions"]

            action = agent.select_action(state, valid_actions)

            obs_next, reward, terminated, _, info_next = env.step(action)
            next_state = tuple(int(x) for x in obs_next)

            # Store transition for episodic (cooperative) learning
            (alice_transitions if agent is alice_agent else bob_transitions).append(
                (state, action, next_state, terminated)
            )

            obs = obs_next
            info = info_next
            total_reward += reward

        # perform learning using the total episodic reward (fully cooperative)
        for s, a, ns, term in alice_transitions:
            alice_agent.update(s, a, total_reward, ns, term)
        for s, a, ns, term in bob_transitions:
            bob_agent.update(s, a, total_reward, ns, term)

        # episode finished
        alice_agent.decay_epsilon()
        bob_agent.decay_epsilon()
        reward_window.append(total_reward)

        if verbose and ep % 1000 == 0:
            avg = np.mean(reward_window[-1000:])
            print(f"  Episode {ep:>6}: avg reward {avg:6.2f}")

    return alice_agent, bob_agent


def evaluate(
    alice: IndependentQLearningAgent,
    bob: IndependentQLearningAgent,
    episodes: int = 100,
) -> float:
    """Run evaluation with ε=0 (greedy). Returns mean episode reward."""
    # Temporarily store & set epsilon to zero for deterministic policy
    prev_eps_alice, prev_eps_bob = alice.cfg.epsilon, bob.cfg.epsilon
    alice.cfg.epsilon = 0.0
    bob.cfg.epsilon = 0.0

    env: gym.Env = CatDogGame()
    returns: List[float] = []

    for _ in range(episodes):
        obs, info = env.reset()
        terminated, ep_return = False, 0.0
        while not terminated:
            state = tuple(int(x) for x in obs)
            agent = alice if env.turn == 0 else bob
            action = agent.select_action(state, info["valid_actions"])
            obs, reward, terminated, _, info = env.step(action)
            ep_return += reward
        returns.append(ep_return)

    # restore exploration rate
    alice.cfg.epsilon, bob.cfg.epsilon = prev_eps_alice, prev_eps_bob
    return float(np.mean(returns))


# ------------------------------------------------------------------------- #
# Cross-play evaluation helpers
# ------------------------------------------------------------------------- #
def cross_play(
    alice_agents: List[IndependentQLearningAgent],
    bob_agents: List[IndependentQLearningAgent],
    episodes: int = 100,
) -> np.ndarray:
    """
    Evaluate every Alice agent with every Bob agent, returning a matrix
    of mean episode returns with shape (len(alice_agents), len(bob_agents)).
    """
    n_a, n_b = len(alice_agents), len(bob_agents)
    matrix = np.zeros((n_a, n_b), dtype=np.float32)
    for i, alice in enumerate(alice_agents):
        for j, bob in enumerate(bob_agents):
            matrix[i, j] = evaluate(alice, bob, episodes=episodes)
    return matrix


# ------------------------------------------------------------------------- #
# Persistence helpers
# ------------------------------------------------------------------------- #
def save_agents(
    alice_agent: IndependentQLearningAgent,
    bob_agent: IndependentQLearningAgent,
    seed_label: str,
) -> None:
    """
    Persist Q-tables to
        <file directory>/trained_agents/seed_{seed_label}/qtable_[alice|bob].pkl
    """
    out_dir = Path(__file__).parent / "trained_agents" / f"seed_{seed_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    alice_agent.save(out_dir / "qtable_alice.pkl")
    bob_agent.save(out_dir / "qtable_bob.pkl")


# ------------------------------------------------------------------------- #
# Command-line interface
# ------------------------------------------------------------------------- #
def main() -> None:
    """
    Run a hard-coded experiment and output results in JSON so that LLMs can
    easily parse them.
    """
    # ------------------------------------------------------------------ #
    # Hyper-parameters (edit here to change behaviour)
    # ------------------------------------------------------------------ #
    EPISODES = 50000      # learning episodes per seed
    EVAL_EPISODES = 500   # evaluation rollouts per pair
    N_SEEDS = 5           # number of independent random seeds
    BASE_SEED = 0         # first seed value (incremented for each run)

    print("=" * 60)
    print("INDEPENDENT Q-LEARNING EXPERIMENT - CAT-DOG COOPERATIVE GAME")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Training episodes per seed: {EPISODES:,}")
    print(f"  - Evaluation episodes per pair: {EVAL_EPISODES}")
    print(f"  - Number of random seeds: {N_SEEDS}")
    print(f"  - Base seed: {BASE_SEED}")
    print()

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    print("TRAINING PHASE")
    print("-" * 20)
    alice_agents: List[IndependentQLearningAgent] = []
    bob_agents: List[IndependentQLearningAgent] = []

    for i in range(N_SEEDS):
        seed = None if BASE_SEED is None else BASE_SEED + i
        print(f"Training seed {seed}...")
        alice, bob = train(EPISODES, seed=seed, verbose=False)
        alice_agents.append(alice)
        bob_agents.append(bob)
        save_agents(alice, bob, str(seed if seed is not None else i))
        
        # Quick evaluation of this seed's self-play performance
        final_reward = evaluate(alice, bob, episodes=100)
        print(f"Seed {seed} final self-play reward: {final_reward:.2f}")
        print()

    # ------------------------------------------------------------------ #
    # Cross-play evaluation
    # ------------------------------------------------------------------ #
    print("CROSS-PLAY EVALUATION")
    print("-" * 25)
    print(f"Evaluating all {N_SEEDS}x{N_SEEDS} agent pairings...")
    print(f"Each pairing evaluated over {EVAL_EPISODES} episodes")
    print()
    
    cp_matrix = cross_play(alice_agents, bob_agents, episodes=EVAL_EPISODES)

    # Calculate self-play (diagonal) and cross-play (off-diagonal) averages
    self_play_rewards = np.diag(cp_matrix)
    
    # Get off-diagonal elements (i != j pairings)
    mask = ~np.eye(cp_matrix.shape[0], dtype=bool)
    cross_play_rewards = cp_matrix[mask]

    result = {
        "cross_play_matrix": cp_matrix.tolist(),
        "avg_self_play_reward": float(np.mean(self_play_rewards)),
        "avg_cross_play_reward": float(np.mean(cross_play_rewards)),
    }

    print("RESULTS SUMMARY")
    print("-" * 15)
    print(f"Average self-play reward (same seed pairings): {result['avg_self_play_reward']:.2f}")
    print(f"Average cross-play reward (different seed pairings): {result['avg_cross_play_reward']:.2f}")
    print(f"Self-play cross-play difference: {result['avg_self_play_reward'] - result['avg_cross_play_reward']:.2f}")
    print()

    # Save results to JSON file in the same directory as the script
    results_path = Path(__file__).parent / "results.json"
    with results_path.open("w") as f:
        json.dump(result, f, indent=2)
    
    print("\nDETAILED RESULTS (JSON):")
    print("=" * 25)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

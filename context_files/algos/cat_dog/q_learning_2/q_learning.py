import numpy as np
from collections import defaultdict
from cat_dog import CatDogGame
import os
import pickle
import json

class TabularIQAgent:
    """
    A simple tabular Q-learning agent for a single agent.
    """
    def __init__(self, action_space_size):
        """
        Initializes the agent.
        
        Args:
            action_space_size (int): The number of possible actions.
        """
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        
    def get_action(self, state, epsilon, valid_actions, rng):
        """
        Chooses an action using an epsilon-greedy policy.
        
        Args:
            state: The current state of the agent.
            epsilon (float): The probability of choosing a random action.
            valid_actions (list): A list of valid actions for the current state.
            
        Returns:
            int: The chosen action.
        """
        if not valid_actions:
            return None # Should not happen in a valid game step

        if rng.random() < epsilon:
            return rng.choice(valid_actions)
        else:
            q_values = self.q_table[state]
            # Select best action among valid actions
            valid_q_values = q_values[valid_actions]
            best_action_index = np.argmax(valid_q_values)
            return valid_actions[best_action_index]

    def update_q_value(self, state, action, reward, next_state_max_q, alpha, gamma):
        """
        Updates the Q-value for a state-action pair.
        
        Args:
            state: The state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state_max_q (float): The maximum Q-value for the next state.
            alpha (float): The learning rate.
            gamma (float): The discount factor.
        """
        old_value = self.q_table[state][action]
        new_value = reward + gamma * next_state_max_q
        self.q_table[state][action] = old_value + alpha * (new_value - old_value)

def train(num_episodes=50000, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.9995, seed=None, verbose=True):
    """
    Trains two independent Q-learning agents (Alice and Bob) on the CatDogGame environment.
    """
    env = CatDogGame()
    rng = np.random.default_rng(seed)
    if seed is not None:
        env.reset(seed=seed)


    alice_agent = TabularIQAgent(action_space_size=env.action_space.n)
    bob_agent = TabularIQAgent(action_space_size=env.action_space.n)

    total_rewards = []
    epsilon = epsilon_start

    for i_episode in range(num_episodes):
        obs, info = env.reset()
        
        # Alice's turn
        alice_state = tuple(obs)
        alice_valid_actions = info['valid_actions']
        alice_action = alice_agent.get_action(alice_state, epsilon, alice_valid_actions, rng)

        next_obs, alice_reward, terminated, _, info = env.step(alice_action)
        
        if terminated: # Alice bailed out
            alice_agent.update_q_value(alice_state, alice_action, alice_reward, 0, alpha, gamma)
        else:
            # Bob's turn
            bob_state = tuple(next_obs)
            bob_valid_actions = info['valid_actions']
            bob_action = bob_agent.get_action(bob_state, epsilon, bob_valid_actions, rng)

            _, bob_reward, terminated, _, info = env.step(bob_action)

            total_reward = alice_reward + bob_reward
            next_state_max_q = 0

            # Update Bob's Q-table (end of episode, next state value is 0)
            bob_agent.update_q_value(bob_state, bob_action, total_reward, 0, alpha, gamma)
            
            # Update Alice's Q-table
            bob_q_values = bob_agent.q_table[bob_state]
            valid_bob_q_values = bob_q_values[bob_valid_actions]
            max_q_bob = np.max(valid_bob_q_values) if valid_bob_q_values.size > 0 else 0
            
            alice_agent.update_q_value(alice_state, alice_action, total_reward, 0, alpha, gamma)

        total_rewards.append(info['total_reward'])
        
        # Decay epsilon
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay
            
        if verbose and (i_episode + 1) % 1000 == 0:
            avg_reward = np.mean(total_rewards[-1000:])
            print(f"TRAINING_PROGRESS: episode={i_episode+1}/{num_episodes} avg_reward_last_1000={avg_reward:.2f} epsilon={epsilon:.3f}")

    if verbose:
        print("\n# Training Results\n")
        print("## Alice's Learned Q-values")
        print("| State (pet) | Signal 1 | Signal 2 | Bail out | Remove barrier |")
        print("|---|---|---|---|---|")
        for state, q_values in sorted(alice_agent.q_table.items()):
            pet = 'cat' if state[0] == 0 else 'dog'
            print(f"| {pet} | {q_values[0]:.2f} | {q_values[1]:.2f} | {q_values[2]:.2f} | {q_values[3]:.2f} |")

        print("\n## Bob's Learned Q-values")
        print("| State (Alice's action, pet) | Bail out | Guess cat | Guess dog |")
        print("|---|---|---|")
        for state, q_values in sorted(bob_agent.q_table.items()):
            pet_visible_val, _, alice_action_val, _ = state
            
            pet_str = "hidden"
            if pet_visible_val != -1:
                pet_str = 'cat' if pet_visible_val == 0 else 'dog'
            
            action_names = ['Signal 1 (light on)', 'Signal 2', 'Bail out', 'Remove barrier']
            alice_action_str = action_names[alice_action_val]

            state_str = f"'{alice_action_str}', {pet_str}"
            print(f"| {state_str} | {q_values[0]:.2f} | {q_values[1]:.2f} | {q_values[2]:.2f} |")

    return alice_agent, bob_agent


def evaluate(alice_agent, bob_agent, env, num_rollouts, seed=None):
    """
    Evaluate a pair of policies (Alice and Bob) over a number of rollouts.
    The policies are used greedily (no exploration).
    """
    rng = np.random.default_rng(seed)
    if seed is not None:
        env.reset(seed=seed)

    total_rewards = []
    for _ in range(num_rollouts):
        obs, info = env.reset()
        
        # Alice's turn
        alice_state = tuple(obs)
        alice_valid_actions = info['valid_actions']
        alice_action = alice_agent.get_action(alice_state, 0.0, alice_valid_actions, rng) # epsilon=0

        next_obs, _, terminated, _, info = env.step(alice_action)
        
        if terminated: # Alice bailed out
            total_rewards.append(info['total_reward'])
        else:
            # Bob's turn
            bob_state = tuple(next_obs)
            bob_valid_actions = info['valid_actions']
            bob_action = bob_agent.get_action(bob_state, 0.0, bob_valid_actions, rng) # epsilon=0

            _, _, terminated, _, info = env.step(bob_action)
            total_rewards.append(info['total_reward'])

    return np.mean(total_rewards)


def main(n_seeds=5, num_episodes_train=50000, num_rollouts_eval=1000):
    """
    Trains multiple agent pairs under different seeds and evaluates them
    using crossplay.
    """
    env = CatDogGame()

    alice_policies = []
    bob_policies = []

    print(f"# Starting Training for {n_seeds} seeds")
    for i in range(n_seeds):
        print(f"\n## Training Seed: {i}")
        # Set verbose=True to get LLM-friendly Q-table printouts
        alice_agent, bob_agent = train(num_episodes=num_episodes_train, seed=i, verbose=False)
        alice_policies.append(alice_agent)
        bob_policies.append(bob_agent)

        # Save Q-tables
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, 'trained_agents', f'seed_{i}')
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, 'alice.pkl'), 'wb') as f:
            pickle.dump(dict(alice_agent.q_table), f)
            
        with open(os.path.join(save_dir, 'bob.pkl'), 'wb') as f:
            pickle.dump(dict(bob_agent.q_table), f)
            
    print("\n# Training Complete")

    crossplay_scores = np.zeros((n_seeds, n_seeds))

    print("\n# Starting Cross-Play Evaluation")
    for i in range(n_seeds):
        for j in range(n_seeds):
            score = evaluate(
                alice_policies[i], 
                bob_policies[j], 
                env, 
                num_rollouts=num_rollouts_eval,
                seed=1000 + i * n_seeds + j # Use a deterministic but unique seed for each eval pair
            )
            crossplay_scores[i, j] = score
            
    print("\n## Cross-play Scores Matrix")
    print("Rows: Alice's policy seed, Columns: Bob's policy seed")
    print("```")
    print(np.around(crossplay_scores, 2))
    print("```\n")
    
    # Calculate metrics
    avg_self_play_reward = np.mean(np.diag(crossplay_scores))

    if n_seeds > 1:
        num_off_diagonal = (n_seeds * n_seeds) - n_seeds
        avg_cross_play_reward = (np.sum(crossplay_scores) - np.sum(np.diag(crossplay_scores))) / num_off_diagonal
    else:
        avg_cross_play_reward = 0.0
    
    results = {
        'cross_play_matrix': crossplay_scores.tolist(),
        'avg_self_play_reward': avg_self_play_reward,
        'avg_cross_play_reward': avg_cross_play_reward,
    }

    print("## Evaluation Metrics")
    print(f"- **Average self-play reward:** {avg_self_play_reward:.2f}")
    print(f"- **Average cross-play reward:** {avg_cross_play_reward:.2f}")
    print(f"- **Self-play / cross-play difference:** {avg_self_play_reward - avg_cross_play_reward:.2f}")

    results_file = os.path.join(script_dir, "results.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    

    return crossplay_scores

if __name__ == '__main__':
    main()

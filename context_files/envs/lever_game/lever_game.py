import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TwoPlayerLeverGame(gym.Env):
    """
    A two-player lever game where both players must choose the same lever to receive a reward.
    
    The game has 10 levers with different reward values. Players receive the lever's reward
    only if both choose the same lever, otherwise they receive 0 reward.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, random_seed=None):
        super().__init__()
        
        # Store random seed for reproducibility
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Reward values for each of the 10 levers
        self.rewards = np.array([1, 1, 1, 1, 1, 0.9, 1, 1, 1, 1])
        self.num_levers = len(self.rewards)
        
        # Action space: both players choose from 10 discrete actions (levers 0-9)
        # Using Tuple space to represent two players' actions
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.num_levers),  # Player 1's action
            spaces.Discrete(self.num_levers)   # Player 2's action
        ))
        
        # Observation space: simple Box containing game state info
        # We'll return [step_count] since this is a one-shot game
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        
        # Game state
        self.done = False
        self.step_count = 0
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Tuple of (player1_action, player2_action)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            raise RuntimeError("Game is already finished. Call reset() to start a new game.")
        
        player1_action, player2_action = action
        
        # Calculate reward: only if both players choose the same lever
        if player1_action == player2_action:
            reward = self.rewards[player1_action]
        else:
            reward = 0.0
        
        # Game terminates after one step
        self.done = True
        self.step_count += 1
        
        # Create observation (just step count normalized)
        observation = np.array([self.step_count], dtype=np.float32)
        
        # Info dictionary with additional information
        info = {
            "player1_action": player1_action,
            "player2_action": player2_action,
            "actions_match": player1_action == player2_action,
            "lever_reward": self.rewards[player1_action] if player1_action == player2_action else None
        }
        
        return observation, reward, True, False, info  # terminated=True, truncated=False
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        self.done = False
        self.step_count = 0
        
        # Initial observation
        observation = np.array([self.step_count], dtype=np.float32)
        
        info = {}
        
        return observation, info
    
    def render(self):
        """
        Render the environment.
        """
        if self.render_mode == "human":
            print(f"Two-Player Lever Game")
            print(f"Lever rewards: {self.rewards}")
            print(f"Step count: {self.step_count}")
            if self.done:
                print("Game finished!")
            else:
                print("Waiting for actions...")
            print("-" * 30)
    
    def close(self):
        """
        Close the environment and clean up resources.
        """
        pass
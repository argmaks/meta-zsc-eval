import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CatDogGame(gym.Env):
    """
    Cat Dog Turn-based Game Environment
    
    Game flow:
    1. Alice observes a random pet (cat=0, dog=1)
    2. Alice chooses from 4 actions:
       - 0: Signal 1 (light on)
       - 1: Signal 2 (light off) 
       - 2: Bail out (fixed reward +1)
       - 3: Remove barrier (cost -5, Bob can see pet directly)
    3. Bob observes Alice's action outcome and chooses from 3 actions:
       - 0: Bail out (fixed reward +0.5)
       - 1: Guess cat
       - 2: Guess dog
    4. If Bob guesses correctly: +10, incorrectly: -10
    """
    
    def __init__(self):
        super().__init__()
        
        # Action spaces
        # Alice: 4 actions, Bob: 3 actions
        # We'll use a single action space and mask invalid actions based on turn
        self.action_space = spaces.Discrete(4)  # Max action space size
        
        # Observation space: [pet_visible, turn, alice_action, game_phase]
        # pet_visible: 0=cat, 1=dog, -1=not visible to current player
        # turn: 0=Alice, 1=Bob  
        # alice_action: 0-3 (Alice's chosen action, -1 if not chosen yet)
        # game_phase: 0=ongoing, 1=ended
        self.observation_space = spaces.Box(
            low=np.array([-1, 0, -1, 0]), 
            high=np.array([1, 1, 3, 1]), 
            dtype=np.int32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random pet assignment
        self.pet = self.np_random.integers(0, 2)  # 0=cat, 1=dog
        self.turn = 0  # 0=Alice's turn, 1=Bob's turn
        self.alice_action = -1  # Alice's action (not chosen yet)
        self.game_phase = 0  # 0=ongoing, 1=ended
        self.total_reward = 0
        
        # Alice can always see the pet, Bob cannot (unless barrier removed)
        pet_visible = self.pet if self.turn == 0 else -1
        
        observation = np.array([pet_visible, self.turn, self.alice_action, self.game_phase], dtype=np.int32)
        info = {
            'pet': 'cat' if self.pet == 0 else 'dog',
            'pet_visible_to_current_player': self.turn == 0,
            'turn': 'Alice' if self.turn == 0 else 'Bob',
            'valid_actions': self._get_valid_actions()
        }
        
        return observation, info
    
    def _get_valid_actions(self):
        """Return list of valid actions for current turn"""
        if self.turn == 0:  # Alice's turn
            return [0, 1, 2, 3]  # All 4 actions available
        else:  # Bob's turn
            return [0, 1, 2]  # Only 3 actions available
    
    def step(self, action):
        if self.game_phase == 1:  # Game already ended
            raise ValueError("Game has already ended. Call reset() to start a new game.")
        
        reward = 0
        terminated = False
        
        if self.turn == 0:  # Alice's turn
            reward = self._alice_action(action)
            if action == 2:  # Alice bailed out
                terminated = True
                self.game_phase = 1
            else:
                self.turn = 1  # Switch to Bob's turn
        else:  # Bob's turn
            reward = self._bob_action(action)
            terminated = True
            self.game_phase = 1
        
        self.total_reward += reward
        
        # Determine pet visibility based on current player and game state
        if self.turn == 0:  # Alice's turn - she can always see the pet
            pet_visible = self.pet
        else:  # Bob's turn - he can only see pet if Alice removed barrier
            pet_visible = self.pet if self.alice_action == 3 else -1
        
        observation = np.array([pet_visible, self.turn, self.alice_action, self.game_phase], dtype=np.int32)
        info = {
            'pet': 'cat' if self.pet == 0 else 'dog',
            'pet_visible_to_current_player': (self.turn == 0) or (self.turn == 1 and self.alice_action == 3),
            'turn': 'Alice' if self.turn == 0 else 'Bob',
            'valid_actions': self._get_valid_actions() if not terminated else [],
            'alice_action': self.alice_action,
            'total_reward': self.total_reward
        }
        
        return observation, reward, terminated, False, info
    
    def _alice_action(self, action):
        """Process Alice's action and return immediate reward"""
        self.alice_action = action
        
        if action == 0:  # Signal 1 (turn on light)
            return 0  # No immediate reward
        elif action == 1:  # Signal 2 (keep light off)
            return 0  # No immediate reward
        elif action == 2:  # Bail out
            return 1  # Fixed reward of 1
        elif action == 3:  # Remove barrier
            return -5  # Cost of 5
        else:
            raise ValueError(f"Invalid action {action} for Alice")
    
    def _bob_action(self, action):
        """Process Bob's action and return reward"""
        if action == 0:  # Bob bails out
            return 0.5  # Fixed reward of 0.5
        elif action == 1:  # Bob guesses cat
            return 10 if self.pet == 0 else -10
        elif action == 2:  # Bob guesses dog
            return 10 if self.pet == 1 else -10
        else:
            raise ValueError(f"Invalid action {action} for Bob")
    
    def render(self, mode='human'):
        """Render the current state of the game"""
        pet_name = 'cat' if self.pet == 0 else 'dog'
        turn_name = 'Alice' if self.turn == 0 else 'Bob'
        
        # Show pet visibility based on current player
        if self.turn == 0:  # Alice's turn
            print(f"Pet (Alice can see): {pet_name}")
        else:  # Bob's turn
            if self.alice_action == 3:  # Barrier was removed
                print(f"Pet (Bob can see because barrier removed): {pet_name}")
            else:
                print("Pet: hidden from Bob")
        
        print(f"Current turn: {turn_name}")
        
        if self.alice_action != -1:
            action_names = ['Signal 1 (light on)', 'Signal 2', 'Bail out', 'Remove barrier']
            print(f"Alice's action: {action_names[self.alice_action]}")
        
        if self.game_phase == 1:
            print(f"Game ended. Total reward: {self.total_reward}")
        
        print("---")
    
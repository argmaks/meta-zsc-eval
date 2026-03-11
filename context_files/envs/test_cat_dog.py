#!/usr/bin/env python3
"""
Test script for the Cat Dog Game environment
"""

import numpy as np
from cat_dog import CatDogGame


def test_basic_functionality():
    """Test basic environment functionality"""
    print("=== Testing Basic Cat Dog Game Functionality ===\n")
    
    env = CatDogGame()
    
    # Test multiple episodes
    for episode in range(3):
        print(f"Episode {episode + 1}:")
        obs, info = env.reset()
        print(f"Initial state: {info}")
        env.render()
        
        total_reward = 0
        step_count = 0
        
        while True:
            # Get valid actions for current turn
            valid_actions = info['valid_actions']
            print(f"Valid actions: {valid_actions}")
            
            # Choose a random valid action
            action = np.random.choice(valid_actions)
            print(f"Chosen action: {action}")
            
            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            print(f"Reward: {reward}, Total reward: {total_reward}")
            env.render()
            
            if terminated or truncated:
                print(f"Episode finished in {step_count} steps with total reward: {total_reward}")
                break
        
        print("\n" + "="*50 + "\n")


def test_specific_scenarios():
    """Test specific game scenarios"""
    print("=== Testing Specific Scenarios ===\n")
    
    env = CatDogGame()
    
    # Scenario 1: Alice bails out immediately
    print("Scenario 1: Alice bails out immediately")
    obs, info = env.reset()
    print(f"Pet: {info['pet']}")
    
    obs, reward, terminated, truncated, info = env.step(2)  # Alice bails out
    print(f"Alice bailed out, reward: {reward}, game ended: {terminated}")
    env.render()
    print()
    
    # Scenario 2: Alice removes barrier, Bob sees pet and guesses correctly
    print("Scenario 2: Alice removes barrier, Bob can see pet")
    obs, info = env.reset()
    pet = info['pet']
    print(f"Pet: {pet}")
    
    # Alice removes barrier
    obs, reward, terminated, truncated, info = env.step(3)
    print(f"Alice removed barrier, immediate reward: {reward}")
    
    # Bob guesses correctly based on seeing the pet
    if pet == 'cat':
        bob_action = 1  # Guess cat
    else:
        bob_action = 2  # Guess dog
    
    obs, reward, terminated, truncated, info = env.step(bob_action)
    print(f"Bob guessed correctly, reward: {reward}, total reward: {info['total_reward']}")
    env.render()
    print()
    
    # Scenario 3: Communication strategy - Alice signals, Bob guesses
    print("Scenario 3: Alice signals, Bob must interpret")
    obs, info = env.reset()
    pet = info['pet']
    print(f"Pet: {pet} (Alice knows this, Bob doesn't)")
    
    # Alice gives a signal (let's say 0 means cat, 1 means dog)
    alice_signal = 0 if pet == 'cat' else 1
    obs, reward, terminated, truncated, info = env.step(alice_signal)
    print(f"Alice gave signal {alice_signal}, immediate reward: {reward}")
    print(f"Bob can see pet: {info['pet_visible_to_current_player']}")
    
    # Bob interprets the signal and guesses (perfect interpretation in this test)
    bob_guess = 1 if alice_signal == 0 else 2  # 1=cat, 2=dog
    obs, reward, terminated, truncated, info = env.step(bob_guess)
    print(f"Bob guessed based on signal, reward: {reward}, total reward: {info['total_reward']}")
    env.render()
    print()
    
    # Scenario 4: Bob guesses without good information
    print("Scenario 4: Alice gives ambiguous signal, Bob must guess blindly")
    obs, info = env.reset()
    pet = info['pet']
    print(f"Pet: {pet} (Alice knows this, Bob doesn't)")
    
    # Alice gives signal 0 regardless of pet (bad strategy)
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"Alice gave signal 0 regardless of pet, immediate reward: {reward}")
    print(f"Bob can see pet: {info['pet_visible_to_current_player']}")
    
    # Bob guesses randomly (50/50 chance)
    bob_guess = np.random.choice([1, 2])  # Random guess
    obs, reward, terminated, truncated, info = env.step(bob_guess)
    guess_name = 'cat' if bob_guess == 1 else 'dog'
    print(f"Bob randomly guessed {guess_name}, reward: {reward}, total reward: {info['total_reward']}")
    env.render()
    print()


def test_action_validation():
    """Test action validation"""
    print("=== Testing Action Validation ===\n")
    
    env = CatDogGame()
    obs, info = env.reset()
    
    print("Testing invalid actions...")
    
    # Test invalid action for Alice
    try:
        env.step(5)  # Invalid action
        print("ERROR: Should have raised an error for invalid action")
    except ValueError as e:
        print(f"✓ Correctly caught invalid Alice action: {e}")
    
    # Take valid Alice action
    obs, reward, terminated, truncated, info = env.step(0)
    
    # Test invalid action for Bob
    try:
        env.step(3)  # Invalid action for Bob (Bob only has 0, 1, 2)
        print("ERROR: Should have raised an error for invalid Bob action")
    except ValueError as e:
        print(f"✓ Correctly caught invalid Bob action: {e}")
    
    print()


if __name__ == "__main__":
    test_basic_functionality()
    test_specific_scenarios() 
    test_action_validation()
    print("All tests completed!") 
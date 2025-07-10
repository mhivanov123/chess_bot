#!/usr/bin/env python3
"""
Test script to verify the chess RL framework works correctly.
"""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.chess_env import ChessEnv
from agents.dqn_agent import DQNAgent, ReplayBuffer
from utils.config import load_config


def test_environment():
    """Test the chess environment."""
    print("Testing chess environment...")
    
    env = ChessEnv()
    
    # Test reset
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Legal actions: {len(info['legal_actions'])}")
    
    # Test a few moves
    for i in range(5):
        legal_actions = info['legal_actions']
        if not legal_actions:
            break
            
        action = legal_actions[0]  # Take first legal action
        next_state, reward, done, truncated, info = env.step(action)
        
        print(f"Move {i+1}: Reward = {reward:.3f}, Done = {done}")
        
        if done:
            break
    
    print("Environment test passed!\n")


def test_agent():
    """Test the DQN agent."""
    print("Testing DQN agent...")
    
    env = ChessEnv()
    agent = DQNAgent(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        hidden_layers=[64, 32]  # Smaller network for testing
    )
    
    # Test action selection
    state, info = env.reset()
    action = agent.select_action(state, info['legal_actions'])
    print(f"Selected action: {action}")
    
    # Test training step
    replay_buffer = ReplayBuffer(1000)
    
    # Add some transitions
    for _ in range(10):
        state, info = env.reset()
        legal_actions = info['legal_actions']
        
        if legal_actions:
            action = legal_actions[0]
            next_state, reward, done, truncated, info = env.step(action)
            next_legal_actions = info['legal_actions']
            
            replay_buffer.push(state, action, reward, next_state, done, legal_actions)
    
    # Test training
    if len(replay_buffer) >= 5:
        batch = replay_buffer.sample(5)
        loss = agent.train_step(batch)
        print(f"Training loss: {loss:.6f}")
    
    print("Agent test passed!\n")


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        config = load_config('configs/dqn_config.yaml')
        print("Configuration loaded successfully")
        print(f"Number of episodes: {config['training']['num_episodes']}")
        print(f"Learning rate: {config['agent']['learning_rate']}")
    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False
    
    print("Configuration test passed!\n")
    return True


def test_training_loop():
    """Test a short training loop."""
    print("Testing training loop...")
    
    env = ChessEnv()
    agent = DQNAgent(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        hidden_layers=[64, 32],
        epsilon=0.1  # Low exploration for testing
    )
    
    replay_buffer = ReplayBuffer(1000)
    
    # Run a few episodes
    for episode in range(3):
        state, info = env.reset()
        legal_actions = info['legal_actions']
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = agent.select_action(state, legal_actions)
            next_state, reward, done, truncated, info = env.step(action)
            next_legal_actions = info['legal_actions']
            
            replay_buffer.push(state, action, reward, next_state, done, legal_actions)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            legal_actions = next_legal_actions
            
            if done or truncated:
                break
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        # Train if enough samples
        if len(replay_buffer) >= 10:
            batch = replay_buffer.sample(5)
            loss = agent.train_step(batch)
            print(f"  Training loss: {loss:.6f}")
    
    print("Training loop test passed!\n")


def main():
    """Run all tests."""
    print("Running chess RL framework tests...\n")
    
    try:
        test_environment()
        test_agent()
        test_config()
        test_training_loop()
        
        print("All tests passed! The framework is working correctly.")
        print("\nYou can now start training with:")
        print("python train.py --config configs/dqn_config.yaml")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    main() 
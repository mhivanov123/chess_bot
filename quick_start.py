#!/usr/bin/env python3
"""
Quick start script for the chess RL framework.
This script demonstrates basic usage and provides a simple training example.
"""

import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.chess_env import ChessEnv
from agents.dqn_agent import DQNAgent, ReplayBuffer
from utils.config import get_default_config


def quick_training_example():
    """
    A quick training example to demonstrate the framework.
    This trains a small agent for a few episodes.
    """
    print("=== Chess RL Framework Quick Start ===\n")
    
    # Create environment
    print("1. Creating chess environment...")
    env = ChessEnv(max_moves=50)  # Shorter games for quick demo
    
    # Create agent with smaller network
    print("2. Creating DQN agent...")
    agent = DQNAgent(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        hidden_layers=[128, 64],  # Smaller network
        learning_rate=0.001,
        epsilon=0.3,  # Higher exploration
        epsilon_decay=0.99
    )
    
    # Create replay buffer
    print("3. Setting up replay buffer...")
    replay_buffer = ReplayBuffer(10000)
    
    # Training parameters
    num_episodes = 20
    min_buffer_size = 100
    batch_size = 32
    
    print(f"4. Starting training for {num_episodes} episodes...")
    print("   (This is a quick demo - real training would use thousands of episodes)\n")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Reset environment
        state, info = env.reset()
        legal_actions = info['legal_actions']
        episode_reward = 0
        episode_length = 0
        
        # Play one episode
        while True:
            # Select action
            action = agent.select_action(state, legal_actions)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            next_legal_actions = info['legal_actions']
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done, legal_actions)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            legal_actions = next_legal_actions
            
            if done or truncated:
                break
        
        # Train if enough samples
        if len(replay_buffer) >= min_buffer_size:
            batch = replay_buffer.sample(batch_size)
            loss = agent.train_step(batch)
            
            # Update target network occasionally
            if episode % 5 == 0:
                agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Print progress
        if episode % 5 == 0:
            print(f"Episode {episode:2d}: Reward = {episode_reward:6.2f}, "
                  f"Length = {episode_length:2d}, Epsilon = {agent.epsilon:.3f}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds!")
    
    # Test the trained agent
    print("\n5. Testing the trained agent...")
    agent.set_training(False)  # Set to evaluation mode
    
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(5):  # Play 5 test games
        state, info = env.reset()
        legal_actions = info['legal_actions']
        
        while True:
            action = agent.select_action(state, legal_actions)
            state, reward, done, truncated, info = env.step(action)
            legal_actions = info['legal_actions']
            
            if done or truncated:
                break
        
        # Determine outcome
        if env.board.is_checkmate():
            if env.board.turn:  # White's turn but checkmated
                losses += 1  # Black won
            else:
                wins += 1    # White won
        else:
            draws += 1  # Draw
    
    print(f"Test results (5 games):")
    print(f"  Wins: {wins}")
    print(f"  Draws: {draws}")
    print(f"  Losses: {losses}")
    
    print("\n=== Quick Start Complete! ===")
    print("\nNext steps:")
    print("1. Run 'python test_framework.py' to verify everything works")
    print("2. Run 'python train.py --config configs/dqn_config.yaml' for full training")
    print("3. Run 'python play.py --model_path models/chess_agent_final.pth' to play against the agent")
    print("4. Check the README.md for more details")


def main():
    """Main function."""
    try:
        quick_training_example()
    except KeyboardInterrupt:
        print("\nQuick start interrupted by user.")
    except Exception as e:
        print(f"Error during quick start: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 
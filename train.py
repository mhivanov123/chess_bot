#!/usr/bin/env python3
"""
Main training script for the chess RL agent.
"""

import argparse
import os
import sys
from src.training.trainer import ChessTrainer
from src.training.ppo_trainer import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description='Train a chess RL agent')
    parser.add_argument('--config', type=str, default='configs/ppo_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--agent', type=str, choices=['dqn', 'ppo'], default='ppo',
                       help='Type of agent to train (default: ppo)')
    parser.add_argument('--plot', action='store_true',
                       help='Plot training curves after training')
    parser.add_argument('--plot_save', type=str, default=None,
                       help='Path to save training curves plot')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Initialize trainer based on agent type
        print(f"Loading configuration from {args.config}")
        
        if args.agent == 'ppo':
            trainer = PPOTrainer(args.config)
        else:
            trainer = ChessTrainer(args.config)
        
        # Start training
        trainer.train()
        
        # Plot training curves if requested
        if args.plot:
            print("Generating training curves...")
            trainer.plot_training_curves(save_path=args.plot_save)
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 
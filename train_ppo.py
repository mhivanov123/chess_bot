#!/usr/bin/env python3
"""
Main training script for the chess PPO agent.
"""

import argparse
import os
import sys
from src.training.ppo_trainer import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description='Train a chess PPO agent')
    parser.add_argument('--config', type=str, default='configs/ppo_config.yaml',
                       help='Path to configuration file')
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
        # Initialize trainer
        print(f"Loading configuration from {args.config}")
        trainer = PPOTrainer(args.config)
        
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
        sys.exit(1)


if __name__ == '__main__':
    main() 
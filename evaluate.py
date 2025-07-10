#!/usr/bin/env python3
"""
Evaluation script for trained chess agents.
"""

import argparse
import os
import sys
import torch
import numpy as np
from src.environment.chess_env import ChessEnv
from src.agents.dqn_agent import DQNAgent


def evaluate_model(model_path: str, num_games: int = 100, render: bool = False):
    """
    Evaluate a trained model by playing games against itself.
    
    Args:
        model_path: Path to the trained model
        num_games: Number of games to play
        render: Whether to render the games
    """
    # Setup environment
    env = ChessEnv()
    
    # Setup agent
    agent = DQNAgent(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n
    )
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    try:
        agent.load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set to evaluation mode
    agent.set_training(False)
    
    # Play games
    wins = 0
    draws = 0
    losses = 0
    total_moves = 0
    game_lengths = []
    
    print(f"Playing {num_games} games for evaluation...")
    
    for game in range(num_games):
        state, info = env.reset()
        legal_actions = info['legal_actions']
        game_moves = 0
        
        if render and game < 5:  # Only render first 5 games
            print(f"\nGame {game + 1}:")
            env.render()
        
        while True:
            action = agent.select_action(state, legal_actions)
            state, reward, done, truncated, info = env.step(action)
            legal_actions = info['legal_actions']
            game_moves += 1
            
            if render and game < 5:
                print(f"Move {game_moves}: {env.board.san(env.board.peek())}")
                env.render()
            
            if done or truncated:
                break
        
        total_moves += game_moves
        game_lengths.append(game_moves)
        
        # Determine game outcome
        if env.board.is_checkmate():
            if env.board.turn:  # White's turn but checkmated
                losses += 1  # Black won
                outcome = "Black wins"
            else:
                wins += 1    # White won
                outcome = "White wins"
        else:
            draws += 1  # Draw
            outcome = "Draw"
        
        if render and game < 5:
            print(f"Game {game + 1} result: {outcome} (moves: {game_moves})")
    
    # Print results
    print(f"\nEvaluation Results ({num_games} games):")
    print(f"Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print(f"Losses: {losses} ({losses/num_games*100:.1f}%)")
    print(f"Average moves per game: {total_moves/num_games:.1f}")
    print(f"Shortest game: {min(game_lengths)} moves")
    print(f"Longest game: {max(game_lengths)} moves")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained chess agent')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--num_games', type=int, default=100,
                       help='Number of games to play for evaluation')
    parser.add_argument('--render', action='store_true',
                       help='Render the first 5 games')
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.num_games, args.render)


if __name__ == '__main__':
    main() 
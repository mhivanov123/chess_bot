#!/usr/bin/env python3
"""
Interactive play script to play against a trained chess agent.
"""

import argparse
import os
import sys
import chess
import torch
import numpy as np
from src.environment.chess_env import ChessEnv
from src.agents.dqn_agent import DQNAgent


def get_user_move(board: chess.Board) -> chess.Move:
    """
    Get a move from the user.
    
    Args:
        board: Current chess board
        
    Returns:
        chess.Move: User's move
    """
    while True:
        try:
            move_str = input("Enter your move (e.g., 'e2e4' or 'Nf3'): ").strip()
            
            if not move_str:
                print("Please enter a move.")
                continue
            
            # Try parsing as UCI format first
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    return move
                else:
                    print("Illegal move. Try again.")
                    continue
            except ValueError:
                pass
            
            # Try parsing as SAN format
            try:
                move = board.parse_san(move_str)
                return move
            except ValueError:
                print("Invalid move format. Use UCI (e.g., 'e2e4') or SAN (e.g., 'Nf3').")
                continue
                
        except KeyboardInterrupt:
            print("\nGame interrupted.")
            sys.exit(0)


def play_game(model_path: str, user_plays_white: bool = True):
    """
    Play a game against the trained agent.
    
    Args:
        model_path: Path to the trained model
        user_plays_white: Whether the user plays as white
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
    
    # Start game
    state, info = env.reset()
    legal_actions = info['legal_actions']
    move_count = 0
    
    print(f"\nStarting new game. You play as {'White' if user_plays_white else 'Black'}.")
    print("The agent plays as {'Black' if user_plays_white else 'White'}.")
    print("Enter moves in UCI format (e.g., 'e2e4') or SAN format (e.g., 'Nf3').")
    print("Type 'quit' to exit.\n")
    
    while True:
        # Display current position
        print(f"\nMove {move_count + 1}")
        print("Current position:")
        env.render()
        print(f"FEN: {env.get_fen()}")
        
        # Check if game is over
        if env.board.is_checkmate():
            winner = "Black" if env.board.turn else "White"
            print(f"\nCheckmate! {winner} wins!")
            break
        elif env.board.is_stalemate():
            print("\nStalemate! The game is a draw.")
            break
        elif env.board.is_insufficient_material():
            print("\nInsufficient material! The game is a draw.")
            break
        
        # Determine whose turn it is
        is_white_turn = env.board.turn
        
        if (is_white_turn and user_plays_white) or (not is_white_turn and not user_plays_white):
            # User's turn
            print("Your turn.")
            
            # Get user move
            try:
                user_input = input("Enter move (or 'quit' to exit): ").strip()
                if user_input.lower() == 'quit':
                    print("Game ended by user.")
                    return
                
                # Parse user move
                try:
                    move = chess.Move.from_uci(user_input)
                except ValueError:
                    try:
                        move = env.board.parse_san(user_input)
                    except ValueError:
                        print("Invalid move format. Try again.")
                        continue
                
                if move not in env.board.legal_moves:
                    print("Illegal move. Try again.")
                    continue
                
                # Apply user move
                env.board.push(move)
                move_count += 1
                
            except KeyboardInterrupt:
                print("\nGame interrupted.")
                return
                
        else:
            # Agent's turn
            print("Agent's turn...")
            
            # Get agent move
            action = agent.select_action(state, legal_actions)
            
            # Apply agent move
            next_state, reward, done, truncated, info = env.step(action)
            legal_actions = info['legal_actions']
            state = next_state
            move_count += 1
            
            # Show agent's move
            last_move = env.board.peek()
            print(f"Agent plays: {env.board.san(last_move)}")


def main():
    parser = argparse.ArgumentParser(description='Play against a trained chess agent')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--black', action='store_true',
                       help='Play as black (agent plays white)')
    
    args = parser.parse_args()
    
    user_plays_white = not args.black
    
    while True:
        play_game(args.model_path, user_plays_white)
        
        play_again = input("\nPlay another game? (y/n): ").strip().lower()
        if play_again not in ['y', 'yes']:
            print("Thanks for playing!")
            break


if __name__ == '__main__':
    main() 
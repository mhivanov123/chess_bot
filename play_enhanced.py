#!/usr/bin/env python3
"""
Enhanced interactive play script with detailed bot analysis.
"""

import argparse
import os
import sys
import chess
import torch
import numpy as np
import time
from typing import List, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.chess_env import ChessEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent


class EnhancedChessPlayer:
    """Enhanced chess player with detailed analysis."""
    
    def __init__(self, model_path: str, agent_type: str = 'ppo'):
        self.env = ChessEnv()
        
        # Load agent
        if agent_type == 'ppo':
            self.agent = PPOAgent(
                state_shape=self.env.observation_space.shape,
                num_actions=self.env.action_space.n
            )
        else:
            self.agent = DQNAgent(
                state_shape=self.env.observation_space.shape,
                num_actions=self.env.action_space.n
            )
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.agent.load(model_path)
        self.agent.set_training(False)
        print(f"Loaded {agent_type.upper()} model from {model_path}")
    
    def get_bot_analysis(self, state: np.ndarray, legal_actions: List[int]) -> Tuple[int, float, float, List[Tuple[str, float]]]:
        """Get detailed bot analysis including top moves."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            
            if isinstance(self.agent, PPOAgent):
                action_probs, value = self.agent.network.get_action_probs(state_tensor, legal_actions)
                action_probs = action_probs.squeeze()
                position_eval = value.item()
            else:  # DQN
                q_values = self.agent.q_network(state_tensor).squeeze()
                # Mask illegal actions
                mask = torch.ones(self.agent.num_actions, device=self.agent.device) * float('-inf')
                mask[legal_actions] = 0
                q_values = q_values + mask
                action_probs = torch.softmax(q_values, dim=0)
                position_eval = q_values.max().item()
            
            # Get top moves
            top_moves = []
            for action_idx in legal_actions:
                if action_idx < len(action_probs):
                    prob = action_probs[action_idx].item()
                    move = self.env._action_to_move(action_idx)
                    move_san = self.env.board.san(move)
                    top_moves.append((move_san, prob))
            
            # Sort by probability
            top_moves.sort(key=lambda x: x[1], reverse=True)
            
            # Get best move
            best_action = action_probs.argmax().item()
            confidence = action_probs[best_action].item()
            
            return best_action, confidence, position_eval, top_moves[:5]
    
    def render_enhanced_board(self, state: np.ndarray, legal_actions: List[int], 
                            show_analysis: bool = True):
        """Render board with enhanced information."""
        print("\n" + "═" * 80)
        print("🎯 ENHANCED CHESS BOARD")
        print("═" * 80)
        
        # Render the board
        print(self.env.board)
        print(f"FEN: {self.env.board.fen()}")
        
        # Game state
        turn = "♔ White" if self.env.board.turn else "♚ Black"
        print(f"Turn: {turn}")
        
        if self.env.board.is_check():
            print("⚠️  CHECK!")
        if self.env.board.is_checkmate():
            print("🏆 CHECKMATE!")
        elif self.env.board.is_stalemate():
            print("🤝 STALEMATE!")
        
        # Show analysis if requested
        if show_analysis and legal_actions:
            try:
                best_action, confidence, position_eval, top_moves = self.get_bot_analysis(state, legal_actions)
                
                print(f"\n🤖 BOT ANALYSIS:")
                print(f"   Position Evaluation: {position_eval:+.3f}")
                print(f"   Best Move: {self.env.board.san(self.env._action_to_move(best_action))}")
                print(f"   Confidence: {confidence:.3f}")
                
                print(f"\n📊 TOP 5 MOVES:")
                for i, (move, prob) in enumerate(top_moves, 1):
                    print(f"   {i}. {move} ({prob:.3f})")
                
            except Exception as e:
                print(f"   Analysis error: {e}")
        
        # Legal moves
        legal_moves_san = [self.env.board.san(move) for move in self.env.board.legal_moves]
        print(f"\n�� Legal moves ({len(legal_moves_san)}): {', '.join(legal_moves_san[:8])}")
        if len(legal_moves_san) > 8:
            print(f"   ... and {len(legal_moves_san) - 8} more")
        
        print("═" * 80)
    
    def get_user_move(self) -> chess.Move:
        """Get a move from the user with enhanced input handling."""
        while True:
            try:
                move_str = input("Enter your move (e.g., 'e2e4', 'Nf3', 'help', 'analysis'): ").strip()
                
                if not move_str:
                    print("Please enter a move.")
                    continue
                
                if move_str.lower() == 'help':
                    print("\n📖 MOVE INPUT HELP:")
                    print("   UCI format: 'e2e4', 'g1f3', 'e7e8q' (for promotion)")
                    print("   SAN format: 'e4', 'Nf3', 'O-O' (castling)")
                    print("   Commands: 'help', 'analysis', 'undo', 'quit'")
                    continue
                
                if move_str.lower() == 'analysis':
                    state = self.env._board_to_state()
                    legal_actions = self.env._get_legal_actions()
                    self.render_enhanced_board(state, legal_actions, show_analysis=True)
                    continue
                
                if move_str.lower() == 'undo':
                    if len(self.env.board.move_stack) > 0:
                        self.env.board.pop()
                        print("Move undone.")
                        return None
                    else:
                        print("No moves to undo.")
                        continue
                
                if move_str.lower() == 'quit':
                    return 'quit'
                
                # Try parsing as UCI format first
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.env.board.legal_moves:
                        return move
                    else:
                        print("❌ Illegal move. Try again.")
                        continue
                except ValueError:
                    pass
                
                # Try parsing as SAN format
                try:
                    move = self.env.board.parse_san(move_str)
                    return move
                except ValueError:
                    print("❌ Invalid move format. Use UCI (e.g., 'e2e4') or SAN (e.g., 'Nf3').")
                    continue
                    
            except KeyboardInterrupt:
                print("\nGame interrupted.")
                return 'quit'
    
    def play_enhanced_game(self, user_plays_white: bool = True):
        """Play an enhanced game with detailed analysis."""
        print(f"\n🎮 ENHANCED CHESS GAME")
        print(f"You play as {'♔ White' if user_plays_white else '♚ Black'}")
        print(f"The bot plays as {'♚ Black' if user_plays_white else '♔ White'}")
        print("Type 'help' for move input help, 'analysis' for bot analysis, 'undo' to undo moves")
        
        state, info = self.env.reset()
        legal_actions = info['legal_actions']
        move_count = 0
        
        while True:
            # Display current position
            self.render_enhanced_board(state, legal_actions, show_analysis=True)
            
            # Check if game is over
            if self.env.board.is_checkmate():
                winner = "♚ Black" if self.env.board.turn else "♔ White"
                print(f"\n�� CHECKMATE! {winner} wins!")
                break
            elif self.env.board.is_stalemate():
                print("\n🤝 STALEMATE! The game is a draw.")
                break
            elif self.env.board.is_insufficient_material():
                print("\n🤝 INSUFFICIENT MATERIAL! The game is a draw.")
                break
            
            # Determine whose turn it is
            is_white_turn = self.env.board.turn
            
            if (is_white_turn and user_plays_white) or (not is_white_turn and not user_plays_white):
                # User's turn
                print("👤 Your turn.")
                
                # Get user move
                user_move = self.get_user_move()
                
                if user_move == 'quit':
                    print("Game ended by user.")
                    return
                
                if user_move is None:  # Undo was called
                    state = self.env._board_to_state()
                    legal_actions = self.env._get_legal_actions()
                    continue
                
                # Apply user move
                self.env.board.push(user_move)
                move_count += 1
                print(f"✅ You played: {self.env.board.san(user_move)}")
                
            else:
                # Bot's turn
                print("🤖 Bot's turn...")
                
                # Get bot analysis and move
                start_time = time.time()
                best_action, confidence, position_eval, top_moves = self.get_bot_analysis(state, legal_actions)
                thinking_time = time.time() - start_time
                
                # Apply bot move
                next_state, reward, done, truncated, info = self.env.step(best_action)
                legal_actions = info['legal_actions']
                state = next_state
                move_count += 1
                
                # Show bot's move and analysis
                last_move = self.env.board.peek()
                move_san = self.env.board.san(last_move)
                print(f"�� Bot plays: {move_san}")
                print(f"   Thinking time: {thinking_time:.2f}s")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Position evaluation: {position_eval:+.3f}")
                
                if done:
                    print("\n🏁 Game Over!")
                    break
            
            print(f"\nMove {move_count} completed. Press Enter to continue...")
            input()


def main():
    parser = argparse.ArgumentParser(description='Enhanced interactive play against chess bot')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--agent_type', type=str, choices=['dqn', 'ppo'], default='ppo',
                       help='Type of agent (default: ppo)')
    parser.add_argument('--black', action='store_true',
                       help='Play as black (bot plays white)')
    
    args = parser.parse_args()
    
    try:
        player = EnhancedChessPlayer(args.model_path, args.agent_type)
        user_plays_white = not args.black
        
        while True:
            player.play_enhanced_game(user_plays_white)
            
            play_again = input("\n�� Play another game? (y/n): ").strip().lower()
            if play_again not in ['y', 'yes']:
                print("�� Thanks for playing!")
                break
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 
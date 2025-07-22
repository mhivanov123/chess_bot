#!/usr/bin/env python3
"""
Comprehensive visualization tools for analyzing chess bot play style.
"""

import argparse
import os
import sys
import chess
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from typing import List, Dict, Tuple, Optional
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.chess_env import ChessEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent


class ChessVisualizer:
    """Comprehensive chess bot visualization tools."""
    
    def __init__(self, model_path: str, agent_type: str = 'ppo'):
        """Initialize visualizer with a trained model."""
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
    
    def render_board_with_analysis(self, state: np.ndarray, legal_actions: List[int], 
                                 move_confidence: Optional[float] = None, 
                                 position_evaluation: Optional[float] = None):
        """Render chess board with analysis information."""
        print("\n" + "="*60)
        print("CHESS POSITION ANALYSIS")
        print("="*60)
        
        # Render the board
        print(self.env.board)
        print(f"FEN: {self.env.board.fen()}")
        
        # Show game state
        if self.env.board.is_check():
            print("⚠️  CHECK!")
        if self.env.board.is_checkmate():
            print("🏆 CHECKMATE!")
        elif self.env.board.is_stalemate():
            print("🤝 STALEMATE!")
        
        # Show turn
        turn = "White" if self.env.board.turn else "Black"
        print(f"Turn: {turn}")
        
        # Show analysis
        if move_confidence is not None:
            print(f"Move Confidence: {move_confidence:.3f}")
        if position_evaluation is not None:
            eval_str = f"{position_evaluation:+.3f}"
            if position_evaluation > 0.5:
                eval_str += " (White advantage)"
            elif position_evaluation < -0.5:
                eval_str += " (Black advantage)"
            else:
                eval_str += " (Equal)"
            print(f"Position Evaluation: {eval_str}")
        
        # Show legal moves
        legal_moves_san = [self.env.board.san(move) for move in self.env.board.legal_moves]
        print(f"Legal moves ({len(legal_moves_san)}): {', '.join(legal_moves_san[:10])}")
        if len(legal_moves_san) > 10:
            print(f"  ... and {len(legal_moves_san) - 10} more")
        
        print("="*60)
    
    def analyze_move_confidence(self, state: np.ndarray, legal_actions: List[int]) -> Tuple[int, float, float]:
        """Analyze the bot's confidence in its move selection."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            
            if isinstance(self.agent, PPOAgent):
                action_probs, value = self.agent.network.get_action_probs(state_tensor, legal_actions)
                action = action_probs.argmax()
                confidence = action_probs[0, action.item()].item()
                position_eval = value.item()
            else:  # DQN
                q_values = self.agent.q_network(state_tensor)
                # Mask illegal actions
                mask = torch.ones(self.agent.num_actions, device=self.agent.device) * float('-inf')
                mask[legal_actions] = 0
                q_values = q_values + mask
                
                action = q_values.argmax()
                max_q = q_values.max().item()
                # Convert Q-value to confidence (normalize)
                confidence = torch.softmax(q_values, dim=1)[0, action.item()].item()
                position_eval = max_q
            
            return action.item(), confidence, position_eval
    
    def interactive_analysis(self):
        """Interactive analysis mode - step through a game with detailed analysis."""
        print("\n🎯 INTERACTIVE ANALYSIS MODE")
        print("This mode lets you step through a game and see the bot's analysis.")
        print("Commands: 'step' (bot moves), 'undo', 'reset', 'quit'")
        
        state, info = self.env.reset()
        legal_actions = info['legal_actions']
        
        while True:
            self.render_board_with_analysis(state, legal_actions)
            
            cmd = input("\nCommand (step/undo/reset/quit): ").strip().lower()
            
            if cmd == 'quit':
                break
            elif cmd == 'reset':
                state, info = self.env.reset()
                legal_actions = info['legal_actions']
            elif cmd == 'undo':
                if len(self.env.board.move_stack) > 0:
                    self.env.board.pop()
                    state = self.env._board_to_state()
                    legal_actions = self.env._get_legal_actions()
            elif cmd == 'step':
                if legal_actions:
                    # Analyze move
                    action, confidence, position_eval = self.analyze_move_confidence(state, legal_actions)
                    
                    # Take the move
                    next_state, reward, done, truncated, info = self.env.step(action)
                    legal_actions = info['legal_actions']
                    state = next_state
                    
                    # Show move analysis
                    last_move = self.env.board.peek()
                    move_san = self.env.board.san(last_move)
                    print(f"\n�� Bot plays: {move_san}")
                    print(f"   Confidence: {confidence:.3f}")
                    print(f"   Position Evaluation: {position_eval:+.3f}")
                    print(f"   Reward: {reward:.3f}")
                    
                    if done:
                        print("\n🏁 Game Over!")
                        break
                else:
                    print("No legal moves available!")
            else:
                print("Unknown command. Use: step, undo, reset, quit")
    
    def analyze_playing_style(self, num_games: int = 50) -> Dict:
        """Analyze the bot's playing style across multiple games."""
        print(f"\n📊 ANALYZING PLAYING STYLE ({num_games} games)...")
        
        # Statistics to collect
        stats = {
            'opening_moves': Counter(),
            'piece_moves': Counter(),
            'move_lengths': [],
            'game_outcomes': Counter(),
            'material_balance': [],
            'move_times': [],
            'position_evaluations': []
        }
        
        for game in range(num_games):
            state, info = self.env.reset()
            legal_actions = info['legal_actions']
            game_moves = []
            game_evaluations = []
            
            while True:
                start_time = time.time()
                
                # Get bot's move and analysis
                action, confidence, position_eval = self.analyze_move_confidence(state, legal_actions)
                
                # Record move time
                move_time = time.time() - start_time
                stats['move_times'].append(move_time)
                stats['position_evaluations'].append(position_eval)
                
                # Take the move
                next_state, reward, done, truncated, info = self.env.step(action)
                legal_actions = info['legal_actions']
                state = next_state
                
                # Record move
                last_move = self.env.board.peek()
                move_san = self.env.board.san(last_move)
                game_moves.append(move_san)
                
                # Record piece moved
                piece = self.env.board.piece_at(last_move.to_square)
                if piece:
                    piece_name = chess.piece_symbol(piece.piece_type).upper()
                    stats['piece_moves'][piece_name] += 1
                
                # Record opening moves (first 10 moves)
                if len(game_moves) <= 10:
                    stats['opening_moves'][move_san] += 1
                
                if done or truncated:
                    break
            
            # Record game statistics
            stats['move_lengths'].append(len(game_moves))
            
            # Determine outcome
            if self.env.board.is_checkmate():
                if self.env.board.turn:  # White's turn but checkmated
                    stats['game_outcomes']['Black wins'] += 1
                else:
                    stats['game_outcomes']['White wins'] += 1
            else:
                stats['game_outcomes']['Draw'] += 1
            
            # Calculate final material balance
            material_balance = self._calculate_material_balance()
            stats['material_balance'].append(material_balance)
            
            if (game + 1) % 10 == 0:
                print(f"  Analyzed {game + 1}/{num_games} games...")
        
        return stats
    
    def _calculate_material_balance(self) -> float:
        """Calculate material balance (positive = white advantage)."""
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        
        white_material = sum(piece_values[piece.piece_type] for square in chess.SQUARES 
                           if (piece := self.env.board.piece_at(square)) and piece.color)
        black_material = sum(piece_values[piece.piece_type] for square in chess.SQUARES 
                           if (piece := self.env.board.piece_at(square)) and not piece.color)
        
        return white_material - black_material
    
    def plot_style_analysis(self, stats: Dict, save_path: Optional[str] = None):
        """Plot comprehensive style analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Chess Bot Playing Style Analysis', fontsize=16, fontweight='bold')
        
        # 1. Game outcomes
        outcomes = list(stats['game_outcomes'].keys())
        counts = list(stats['game_outcomes'].values())
        axes[0, 0].pie(counts, labels=outcomes, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Game Outcomes')
        
        # 2. Move lengths distribution
        axes[0, 1].hist(stats['move_lengths'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Game Length Distribution')
        axes[0, 1].set_xlabel('Number of Moves')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Piece movement preferences
        pieces = list(stats['piece_moves'].keys())
        piece_counts = list(stats['piece_moves'].values())
        axes[0, 2].bar(pieces, piece_counts, color='lightgreen')
        axes[0, 2].set_title('Piece Movement Preferences')
        axes[0, 2].set_ylabel('Number of Moves')
        
        # 4. Opening move preferences (top 10)
        top_openings = stats['opening_moves'].most_common(10)
        opening_moves = [move for move, _ in top_openings]
        opening_counts = [count for _, count in top_openings]
        axes[1, 0].barh(range(len(opening_moves)), opening_counts, color='orange')
        axes[1, 0].set_yticks(range(len(opening_moves)))
        axes[1, 0].set_yticklabels(opening_moves)
        axes[1, 0].set_title('Top 10 Opening Moves')
        axes[1, 0].set_xlabel('Frequency')
        
        # 5. Position evaluation over time
        if stats['position_evaluations']:
            # Sample evaluations (every 10th evaluation to avoid overcrowding)
            sample_evaluations = stats['position_evaluations'][::10]
            axes[1, 1].plot(sample_evaluations, alpha=0.7, color='purple')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Position Evaluations (Sample)')
            axes[1, 1].set_xlabel('Move Index (x10)')
            axes[1, 1].set_ylabel('Evaluation')
        
        # 6. Move timing distribution
        if stats['move_times']:
            axes[1, 2].hist(stats['move_times'], bins=20, alpha=0.7, color='red', edgecolor='black')
            axes[1, 2].set_title('Move Timing Distribution')
            axes[1, 2].set_xlabel('Time (seconds)')
            axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Style analysis saved to {save_path}")
        else:
            plt.show()
    
    def create_move_heatmap(self, num_games: int = 20) -> np.ndarray:
        """Create a heatmap showing where the bot likes to move pieces from/to."""
        print(f"\n��️  CREATING MOVE HEATMAP ({num_games} games)...")
        
        # Initialize heatmaps for each piece type
        piece_heatmaps = {
            'P': np.zeros((8, 8)),  # Pawns
            'N': np.zeros((8, 8)),  # Knights
            'B': np.zeros((8, 8)),  # Bishops
            'R': np.zeros((8, 8)),  # Rooks
            'Q': np.zeros((8, 8)),  # Queens
            'K': np.zeros((8, 8))   # Kings
        }
        
        for game in range(num_games):
            state, info = self.env.reset()
            legal_actions = info['legal_actions']
            
            while True:
                action, _, _ = self.analyze_move_confidence(state, legal_actions)
                next_state, reward, done, truncated, info = self.env.step(action)
                legal_actions = info['legal_actions']
                state = next_state
                
                # Record the move
                last_move = self.env.board.peek()
                piece = self.env.board.piece_at(last_move.to_square)
                if piece:
                    piece_symbol = chess.piece_symbol(piece.piece_type).upper()
                    if piece_symbol in piece_heatmaps:
                        # Record the destination square
                        rank, file = chess.square_rank(last_move.to_square), chess.square_file(last_move.to_square)
                        piece_heatmaps[piece_symbol][rank, file] += 1
                
                if done or truncated:
                    break
            
            if (game + 1) % 5 == 0:
                print(f"  Processed {game + 1}/{num_games} games...")
        
        return piece_heatmaps
    
    def plot_move_heatmaps(self, piece_heatmaps: Dict[str, np.ndarray], save_path: Optional[str] = None):
        """Plot move heatmaps for each piece type."""
        piece_names = {'P': 'Pawns', 'N': 'Knights', 'B': 'Bishops', 
                      'R': 'Rooks', 'Q': 'Queens', 'K': 'Kings'}
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Chess Bot Move Heatmaps by Piece Type', fontsize=16, fontweight='bold')
        
        for i, (piece, heatmap) in enumerate(piece_heatmaps.items()):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            # Create heatmap
            sns.heatmap(heatmap, annot=True, fmt='.0f', cmap='YlOrRd', 
                       square=True, cbar_kws={'label': 'Move Count'}, ax=ax)
            ax.set_title(f'{piece_names[piece]} Move Heatmap')
            ax.set_xlabel('File (a-h)')
            ax.set_ylabel('Rank (1-8)')
            
            # Add chess coordinates
            ax.set_xticks(np.arange(8) + 0.5)
            ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
            ax.set_yticks(np.arange(8) + 0.5)
            ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Move heatmaps saved to {save_path}")
        else:
            plt.show()
    
    def replay_game_with_analysis(self, game_moves: List[str]):
        """Replay a game with detailed analysis at each move."""
        print("\n🎬 GAME REPLAY WITH ANALYSIS")
        print("="*60)
        
        state, info = self.env.reset()
        legal_actions = info['legal_actions']
        
        for i, move_san in enumerate(game_moves):
            print(f"\nMove {i+1}: {move_san}")
            print("-" * 40)
            
            # Show position before move
            self.render_board_with_analysis(state, legal_actions)
            
            # Analyze bot's thinking
            action, confidence, position_eval = self.analyze_move_confidence(state, legal_actions)
            
            print(f"\n�� Bot's Analysis:")
            print(f"   Selected move: {self.env.board.san(self.env._action_to_move(action))}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Position evaluation: {position_eval:+.3f}")
            
            # Apply the move
            next_state, reward, done, truncated, info = self.env.step(action)
            legal_actions = info['legal_actions']
            state = next_state
            
            if done:
                print(f"\n🏁 Game Over! Final result: {reward:.3f}")
                break
            
            input("\nPress Enter to continue to next move...")


def main():
    parser = argparse.ArgumentParser(description='Visualize chess bot playing style')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--agent_type', type=str, choices=['dqn', 'ppo'], default='ppo',
                       help='Type of agent (default: ppo)')
    parser.add_argument('--mode', type=str, 
                       choices=['interactive', 'style_analysis', 'heatmap', 'replay'],
                       default='interactive',
                       help='Visualization mode')
    parser.add_argument('--num_games', type=int, default=50,
                       help='Number of games for analysis')
    parser.add_argument('--save_plots', type=str, default=None,
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    try:
        visualizer = ChessVisualizer(args.model_path, args.agent_type)
        
        if args.mode == 'interactive':
            visualizer.interactive_analysis()
        
        elif args.mode == 'style_analysis':
            stats = visualizer.analyze_playing_style(args.num_games)
            save_path = os.path.join(args.save_plots, 'style_analysis.png') if args.save_plots else None
            visualizer.plot_style_analysis(stats, save_path)
            
            # Print summary statistics
            print(f"\n📈 STYLE ANALYSIS SUMMARY:")
            print(f"Average game length: {np.mean(stats['move_lengths']):.1f} moves")
            print(f"Most common piece moved: {stats['piece_moves'].most_common(1)[0]}")
            print(f"Average move time: {np.mean(stats['move_times']):.3f} seconds")
            print(f"Average material balance: {np.mean(stats['material_balance']):.1f}")
        
        elif args.mode == 'heatmap':
            piece_heatmaps = visualizer.create_move_heatmap(args.num_games)
            save_path = os.path.join(args.save_plots, 'move_heatmaps.png') if args.save_plots else None
            visualizer.plot_move_heatmaps(piece_heatmaps, save_path)
        
        elif args.mode == 'replay':
            # Example game replay (you can modify this to replay specific games)
            print("Replay mode - you can modify the script to replay specific games")
            # Example: visualizer.replay_game_with_analysis(['e4', 'e5', 'Nf3', 'Nc6'])
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 
import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List
import random


class ChessEnv(gym.Env):
    """
    Chess environment for reinforcement learning.
    Uses full action space with legal move masking.
    """
    
    def __init__(self, max_moves: int = 200, reward_material_factor: float = 0.1):
        super().__init__()
        
        self.max_moves = max_moves
        self.reward_material_factor = reward_material_factor
        self.board = None
        self.move_count = 0
        
        # Full action space - all possible moves (from_square * to_square + promotions)
        self.num_actions = 64 * 64 + 64 * 4  # 4096 + 256 = 4352
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, 8, 12), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_actions)
        
    def _board_to_state(self) -> np.ndarray:
        """Convert chess board to neural network input state."""
        state = np.zeros((8, 8, 12), dtype=np.float32)
        
        piece_channels = {
            chess.PAWN: (0, 6),
            chess.KNIGHT: (1, 7),
            chess.BISHOP: (2, 8),
            chess.ROOK: (3, 9),
            chess.QUEEN: (4, 10),
            chess.KING: (5, 11)
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                rank, file = chess.square_rank(square), chess.square_file(square)
                piece_type = piece.piece_type
                color = piece.color
                
                white_channel, black_channel = piece_channels[piece_type]
                channel = white_channel if color else black_channel
                state[rank, file, channel] = 1.0
        
        return state
    
    def _action_to_move(self, action: int) -> chess.Move:
        """Convert action index to chess move."""
        if action < 4096:  # Regular moves
            from_square = action // 64
            to_square = action % 64
            return chess.Move(from_square, to_square)
        else:  # Promotion moves
            action = action - 4096
            from_square = action // 4
            promotion_piece = action % 4 + 1  # QUEEN=1, ROOK=2, BISHOP=3, KNIGHT=4
            to_square = from_square + (8 if from_square < 56 else -8)  # Move forward/backward
            return chess.Move(from_square, to_square, promotion_piece)
    
    def _move_to_action(self, move: chess.Move) -> int:
        """Convert chess move to action index."""
        if move.promotion:
            return 4096 + move.from_square * 4 + (move.promotion - 1)
        else:
            return move.from_square * 64 + move.to_square
    
    def _get_legal_actions(self) -> List[int]:
        """Get list of legal action indices."""
        return [self._move_to_action(move) for move in self.board.legal_moves]
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on game state."""
        if self.board.is_checkmate():
            return -1.0 if self.board.turn else 1.0
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0
        elif self.move_count >= self.max_moves:
            return 0.0
        
        # Material advantage
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        
        white_material = sum(piece_values[piece.piece_type] for square in chess.SQUARES 
                           if (piece := self.board.piece_at(square)) and piece.color)
        black_material = sum(piece_values[piece.piece_type] for square in chess.SQUARES 
                           if (piece := self.board.piece_at(square)) and not piece.color)
        
        return (white_material - black_material) * self.reward_material_factor
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.board = chess.Board()
        self.move_count = 0
        
        state = self._board_to_state()
        info = {
            'legal_actions': self._get_legal_actions(),
            'legal_moves_san': [self.board.san(move) for move in self.board.legal_moves]
        }
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Convert action to move
        move = self._action_to_move(action)
        
        # Check if move is legal
        if move not in self.board.legal_moves:
            return self._board_to_state(), -10.0, True, False, {
                'illegal_move': True,
                'legal_actions': self._get_legal_actions(),
                'last_move_san': None
            }
        
        # Apply move
        self.board.push(move)
        self.move_count += 1
        
        # Check if game is over
        terminated = (self.board.is_checkmate() or self.board.is_stalemate() or 
                     self.board.is_insufficient_material() or self.move_count >= self.max_moves)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get state and info
        state = self._board_to_state()
        
        # Safely get the last move SAN
        last_move_san = None
        if self.board.move_stack:
            try:
                last_move_san = self.board.san(self.board.peek())
            except (AssertionError, ValueError):
                # Handle case where peek() returns an invalid move
                last_move_san = None
        
        info = {
            'legal_actions': self._get_legal_actions(),
            'legal_moves_san': [self.board.san(move) for move in self.board.legal_moves],
            'last_move_san': last_move_san
        }
        
        return state, reward, terminated, False, info
    
    def render(self):
        """Render the current board state."""
        print(self.board)
        print(f"Legal moves: {[self.board.san(move) for move in self.board.legal_moves]}") 
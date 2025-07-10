import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import random


class ChessEnv(gym.Env):
    """
    Chess environment for reinforcement learning.
    Implements the Gymnasium interface.
    """
    
    def __init__(self, max_moves: int = 200, reward_material_factor: float = 0.1):
        super().__init__()
        
        self.max_moves = max_moves
        self.reward_material_factor = reward_material_factor
        self.board = None
        self.move_count = 0
        self.legal_moves = []
        self.move_to_action = {}
        self.action_to_move = {}
        
        # Initialize action space mapping
        self._init_action_space()
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, 8, 12), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.move_to_action))
        
    def _init_action_space(self):
        """Initialize the mapping between moves and action indices."""
        # Generate all possible moves (from square, to square)
        moves = []
        for from_square in chess.SQUARES:
            for to_square in chess.SQUARES:
                if from_square != to_square:
                    moves.append((from_square, to_square))
        
        # Add promotion moves
        for from_square in range(8, 16):  # Pawns on second rank
            for to_square in range(0, 8):  # Promotion squares
                for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    moves.append((from_square, to_square, piece))
        
        # Create mappings
        for i, move in enumerate(moves):
            self.move_to_action[move] = i
            self.action_to_move[i] = move
    
    def _board_to_state(self) -> np.ndarray:
        """
        Convert chess board to neural network input state.
        Returns a 8x8x12 tensor representing piece positions and types.
        """
        state = np.zeros((8, 8, 12), dtype=np.float32)
        
        # Piece types: pawn, knight, bishop, rook, queen, king
        # For each piece type, we have white and black versions
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
                
                # Determine channel index
                white_channel, black_channel = piece_channels[piece_type]
                channel = white_channel if color else black_channel
                
                state[rank, file, channel] = 1.0
        
        return state
    
    def _get_legal_actions(self) -> list:
        """Get list of legal action indices."""
        legal_actions = []
        for move in self.board.legal_moves:
            # Convert move to action index
            from_square = move.from_square
            to_square = move.to_square
            promotion = move.promotion if move.promotion else None
            
            if promotion:
                action_key = (from_square, to_square, promotion)
            else:
                action_key = (from_square, to_square)
            
            if action_key in self.move_to_action:
                legal_actions.append(self.move_to_action[action_key])
        
        return legal_actions
    
    def _apply_action(self, action: int) -> bool:
        """
        Apply action to the board.
        Returns True if move was legal, False otherwise.
        """
        if action not in self.action_to_move:
            return False
        
        move_tuple = self.action_to_move[action]
        
        # Create chess.Move object
        if len(move_tuple) == 3:  # Promotion move
            from_square, to_square, promotion = move_tuple
            move = chess.Move(from_square, to_square, promotion)
        else:  # Regular move
            from_square, to_square = move_tuple
            move = chess.Move(from_square, to_square)
        
        # Check if move is legal
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on game state.
        """
        if self.board.is_checkmate():
            # Checkmate - reward based on who won
            if self.board.turn:  # White's turn but checkmated
                return -1.0  # Black won
            else:
                return 1.0   # White won
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0  # Draw
        elif self.move_count >= self.max_moves:
            return 0.0  # Draw by move limit
        
        # Material advantage reward
        material_diff = self._get_material_advantage()
        return material_diff * self.reward_material_factor
    
    def _get_material_advantage(self) -> float:
        """
        Calculate material advantage for white.
        Positive values favor white, negative favor black.
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King has no material value
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                value = piece_values[piece.piece_type]
                if piece.color:  # White
                    white_material += value
                else:  # Black
                    black_material += value
        
        return white_material - black_material
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.board = chess.Board()
        self.move_count = 0
        
        state = self._board_to_state()
        info = {
            'legal_actions': self._get_legal_actions(),
            'material_advantage': self._get_material_advantage()
        }
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index to take
            
        Returns:
            observation: Current board state
            reward: Reward for the action
            terminated: Whether episode is terminated
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Apply action
        legal = self._apply_action(action)
        
        if not legal:
            # Illegal move - penalize heavily
            return self._board_to_state(), -10.0, True, False, {'illegal_move': True}
        
        self.move_count += 1
        
        # Check if game is over
        terminated = (
            self.board.is_checkmate() or
            self.board.is_stalemate() or
            self.board.is_insufficient_material() or
            self.move_count >= self.max_moves
        )
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get current state and info
        state = self._board_to_state()
        info = {
            'legal_actions': self._get_legal_actions(),
            'material_advantage': self._get_material_advantage(),
            'move_count': self.move_count,
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate()
        }
        
        return state, reward, terminated, False, info
    
    def render(self):
        """Render the current board state."""
        print(self.board)
    
    def get_fen(self) -> str:
        """Get FEN string representation of current board."""
        return self.board.fen() 
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
    
    def __init__(self, max_moves: int = 200, reward_material_factor: float = 0.1,
             reward_win: float = 100.0, reward_draw: float = 0.0, reward_loss: float = -100.0):
        super().__init__()
        
        self.max_moves = max_moves
        self.reward_material_factor = reward_material_factor
        self.reward_win = reward_win
        self.reward_draw = reward_draw
        self.reward_loss = reward_loss
        self.board = None
        self.move_count = 0
        
        # Full action space: 64*64 for all from-to, plus 64*3*2 for all possible promotions (excluding underpromotions to king)
        self.num_actions = 64 * 64 + 64 * 3 * 2  # 4096 + 384 = 4480
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, 8, 13), dtype=np.float32  # 13 channels now
        )
        self.action_space = spaces.Discrete(self.num_actions)
        
    def _board_to_state(self) -> np.ndarray:
        """Convert chess board to neural network input state."""
        state = np.zeros((8, 8, 13), dtype=np.float32)  # 13 channels now
        
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
        # Add turn channel: 1.0 if white to move, 0.0 if black to move
        state[:, :, 12] = 0.0 if self.board.turn == chess.WHITE else 1.0


        
        return state
    
    def _action_to_move(self, action: int) -> chess.Move:
        """Convert action index to chess move."""
        #print(action)
        if action < 4096:
            from_square = action // 64
            to_square = action % 64
            return chess.Move(from_square, to_square)
        else:
            # Promotion moves: enumerate all possible pawn promotions (excluding king)
            # 384 promotion actions: 64 squares * 3 underpromotions * 2 colors
            promo_action = action - 4096
            from_square = promo_action // 6
            promo_type_idx = (promo_action % 6)
            # 0: Q, 1: R, 2: B, 3: N (white), 4: Q, 5: R, 6: B, 7: N (black)
            if promo_type_idx < 3:
                color = chess.WHITE
                promotion = [chess.QUEEN, chess.ROOK, chess.BISHOP][promo_type_idx]
                to_square = from_square + 8
            else:
                color = chess.BLACK
                promotion = [chess.QUEEN, chess.ROOK, chess.BISHOP][promo_type_idx - 3]
                to_square = from_square - 8
            return chess.Move(from_square, to_square, promotion=promotion)
    
    def _move_to_action(self, move: chess.Move) -> int:
        """Convert chess move to action index."""
        if move.promotion:
            # Only handle Q, R, B promotions for both colors
            promo_map = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2}
            if move.from_square < 56:  # White promotion
                promo_type_idx = promo_map[move.promotion]
            else:  # Black promotion
                promo_type_idx = promo_map[move.promotion] + 3
            return 4096 + move.from_square * 6 + promo_type_idx
        else:
            return move.from_square * 64 + move.to_square
    
    def _get_legal_actions(self) -> List[int]:
        """Get list of legal action indices."""
        actions = []
        for move in self.board.legal_moves:
            try:
                action = self._move_to_action(move)
                # Validate round-trip
                if self._action_to_move(action) == move:
                    actions.append(action)
            except Exception:
                continue
        return actions
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current position."""
        # Game ending conditions
        if self.board.is_checkmate():
            return self.reward_win

        else:
            return 0.0
        
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return self.reward_draw
        
        # Material advantage - symmetric for both colors
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # Usually 0, since the king can't be captured
        }
        white_material = sum(piece_values[piece.piece_type] for square in chess.SQUARES 
                           if (piece := self.board.piece_at(square)) and piece.color)
        black_material = sum(piece_values[piece.piece_type] for square in chess.SQUARES 
                           if (piece := self.board.piece_at(square)) and not piece.color)
        
        material_diff = white_material - black_material
        
        # If it's White's turn, positive material_diff is good for White
        # If it's Black's turn, negative material_diff is good for Black
        # So we need to flip the sign based on whose turn it is
        if not self.board.turn:  # Black's turn
            material_diff = -material_diff
        
        return material_diff * self.reward_material_factor
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.board = chess.Board()
        self.move_count = 0
        
        state = self._board_to_state()
        
        # Validate action space mapping
        if not self._validate_action_space():
            raise ValueError("Action space mapping is inconsistent!")
        
        info = {
            'legal_actions': self._get_legal_actions(),
            'legal_moves_san': [self.board.san(move) for move in self.board.legal_moves]
        }
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        try:
            move = self._action_to_move(action)
            # Check that move squares are in bounds (0-63)
            if not (0 <= move.from_square < 64 and 0 <= move.to_square < 64):
                raise ValueError(f"Move squares out of bounds: from {move.from_square}, to {move.to_square}")
        except Exception as e:
            # Action could not be converted to a valid move (invalid index, etc.)
            return self._board_to_state(), -10.0, True, False, {
                'illegal_move': True,
                'legal_actions': self._get_legal_actions(),
                'last_move_san': None,
                'error': str(e)
            }
        # Now you can safely check legality
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

    def _validate_action_space(self):
        """Validate that action space mapping is consistent."""
        print("Validating action space mapping...")
        
        # Test all legal moves in current position
        legal_moves = list(self.board.legal_moves)
        legal_actions = self._get_legal_actions()
        
        print(f"Legal moves: {len(legal_moves)}")
        print(f"Legal actions: {len(legal_actions)}")
        
        # Test round-trip conversion for each legal move
        for move in legal_moves:
            action = self._move_to_action(move)
            reconstructed_move = self._action_to_move(action)
            
            if move != reconstructed_move:
                print(f"❌ Mapping error: {move} -> {action} -> {reconstructed_move}")
                return False
        
        print("✅ Action space mapping is consistent!")
        return True 
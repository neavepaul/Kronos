import numpy as np
import chess
from typing import Tuple, List, Dict
import logging

# Configure logging
logger = logging.getLogger('athena.utils')

def fen_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert board position to input tensor format (8x8x20)."""
    logger.debug(f"Converting board to tensor: {board.fen()}")
    tensor = np.zeros((8, 8, 20), dtype=np.float32)
    
    # Piece planes (12 layers: 6 piece types x 2 colors)
    piece_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Fill piece planes
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = divmod(square, 8)
            tensor[rank, file, piece_idx[piece.symbol()]] = 1
            
    # Additional feature planes (8 layers)
    current_player = int(board.turn)
    total_moves = board.fullmove_number
    
    # Repetition count (up to 2)
    rep_count = min(board.is_repetition(2), 2)
    
    # Fill auxiliary planes
    tensor[:, :, 12] = current_player  # Current player to move
    tensor[:, :, 13] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[:, :, 14] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[:, :, 15] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[:, :, 16] = float(board.has_queenside_castling_rights(chess.BLACK))
    tensor[:, :, 17] = rep_count / 2.0  # Normalized repetition count
    tensor[:, :, 18] = min(total_moves / 100.0, 1.0)  # Normalized move count
    tensor[:, :, 19] = board.halfmove_clock / 100.0  # Normalized fifty-move counter
    
    return tensor

def get_attack_defense_maps(board: chess.Board) -> Tuple[np.ndarray, np.ndarray]:
    """Generate attack and defense maps for the current position."""
    logger.debug(f"Generating attack/defense maps for position: {board.fen()}")
    attack_map = np.zeros((8, 8), dtype=np.float32)
    defense_map = np.zeros((8, 8), dtype=np.float32)
    
    # For each square
    for square in chess.SQUARES:
        rank, file = divmod(square, 8)
        
        # Count attackers and defenders
        attackers = board.attackers(not board.turn, square)
        defenders = board.attackers(board.turn, square)
        
        attack_map[rank, file] = len(attackers)
        defense_map[rank, file] = len(defenders)
    
    # Normalize maps
    attack_map = attack_map / 8.0  # Maximum 8 pieces can attack a square
    defense_map = defense_map / 8.0
    
    return attack_map, defense_map

def encode_move_sequence(moves: List[str], max_length: int = 50) -> np.ndarray:
    """Encode a sequence of moves for the history input."""
    logger.debug(f"Encoding move sequence, length: {len(moves)}")
    # Each move is encoded as: from_square (6 bits) + to_square (6 bits) = 12 bits
    encoded = np.zeros(max_length, dtype=np.float32)
    
    for i, move in enumerate(moves[-max_length:]):  # Keep last max_length moves
        if i >= max_length:
            break
            
        # Convert UCI move string to indices
        from_square = chess.parse_square(move[:2])
        to_square = chess.parse_square(move[2:4])
        
        # Normalize square indices
        encoded[i] = (from_square * 64 + to_square) / 4095.0  # Normalize to [0, 1]
    
    return encoded

def move_to_index(move: chess.Move) -> int:
    """Convert a chess move to a single index (0-4095)."""
    return move.from_square * 64 + move.to_square

def index_to_move(index: int) -> chess.Move:
    """Convert an index back to a chess move."""
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)

def encode_move(move: chess.Move) -> np.ndarray:
    """One-hot encode a chess move."""
    encoding = np.zeros(4096, dtype=np.float32)  # 64*64 possible moves
    encoding[move_to_index(move)] = 1.0
    return encoding

def decode_move(encoding: np.ndarray) -> chess.Move:
    """Convert one-hot move encoding back to a chess move."""
    index = np.argmax(encoding)
    return index_to_move(index)

def create_move_lookup() -> Dict[str, int]:
    """Create a lookup table for move strings to indices."""
    lookup = {}
    for from_square in range(64):
        for to_square in range(64):
            move = chess.Move(from_square, to_square)
            lookup[move.uci()] = move_to_index(move)
    return lookup
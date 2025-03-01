import json
import os
import h5py
import sqlite3
import chess
import chess.engine
import numpy as np
import tensorflow as tf
import random
import requests

HDF5_FILE = "training_data/training_data.hdf5"
MAX_MOVE_HISTORY = 50  # Fixed-length move history
STOCKFISH_API = "https://stockfish.online/api/s/v2.php"

STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
EVAL_DB_PATH = "eval_cache.sqlite"


# Convert FEN to Tensor Representation
def fen_to_tensor(fen):
    """
    Converts a FEN string into a 8x8x20 tensor representation.
    """
    piece_mapping = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
    }

    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    parts = fen.split()
    piece_placement, active_color, castling_rights, en_passant, halfmove, fullmove = parts

    # Process piece placements
    rows = piece_placement.split("/")
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                tensor[i, col, piece_mapping[char]] = 1.0
                col += 1

    external_channels = np.zeros((8, 8, 8), dtype=np.float32)

    # Active Color (White = 1, Black = 0)
    external_channels[:, :, 0] = 1.0 if active_color == "w" else 0.0

    # Castling Rights
    castling_flags = ["K", "Q", "k", "q"]
    for j, flag in enumerate(castling_flags):
        if flag in castling_rights:
            external_channels[:, :, j + 1] = 1.0

    # En Passant Square
    if en_passant != "-":
        col = ord(en_passant[0]) - ord("a")
        row = 8 - int(en_passant[1])  
        external_channels[row, col, 5] = 1.0  

    # Normalize Clocks
    external_channels[:, :, 6] = float(halfmove) / 50.0
    external_channels[:, :, 7] = float(fullmove) / 500.0

    full_tensor = np.concatenate((tensor, external_channels), axis=-1)  
    return full_tensor

def init_eval_db():
    conn = sqlite3.connect(EVAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS eval_cache (
            fen TEXT PRIMARY KEY,
            evaluation REAL
        )
    ''')
    conn.commit()
    conn.close()

def get_cached_eval(fen):
    conn = sqlite3.connect(EVAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT evaluation FROM eval_cache WHERE fen = ?", (fen,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return None

def store_eval(fen, evaluation):
    conn = sqlite3.connect(EVAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO eval_cache (fen, evaluation) VALUES (?, ?)", (fen, evaluation))
    conn.commit()
    conn.close()


def get_stockfish_eval(fen, depth=12):
    """
    Fetches the Stockfish evaluation for a given position via API.

    Returns:
        float: Evaluation score in pawns.
               - Positive = White is better
               - Negative = Black is better
               - ±1000 for checkmate situations
               - 0 if API fails
    """
    try:
        response = requests.get(STOCKFISH_API, params={"fen": fen, "depth": depth})
        data = response.json()

        if not data.get("success"):
            return 0  # Default if API call fails
        
        if "mate" in data and data["mate"] is not None:
            return 1000 if data["mate"] > 0 else -1000  # Checkmate detected
        
        return data.get("evaluation", 0) / 100  # Convert centipawn to pawns
    except Exception as e:
        print(f"⚠️ Stockfish API Error: {e}")
        return 0  # Return neutral eval if API fails


def get_stockfish_eval_train(fen, stockfish_path=STOCKFISH_PATH, depth=12):
    """
    Fetches the Stockfish evaluation for a given position using local Stockfish engine.
    """
    try:
        cached_eval = get_cached_eval(fen)
        if cached_eval is not None:
            return cached_eval

        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            eval_score = info["score"].relative

            if eval_score.is_mate():
                value = 1000 if eval_score.mate() > 0 else -1000
            else:
                value = eval_score.score() / 100

        store_eval(fen, value)
        return value
            
    except Exception as e:
        print(f"⚠️ Stockfish Local Eval Error: {e}")
        return 0  # Return neutral eval if engine call fails

def get_attack_defense_maps(board):
    """
    Computes attack and defense maps for each square on the board.
    Returns:
        attack_map (8x8): Number of pieces attacking each square.
        defense_map (8x8): Number of pieces defending each square.
    """
    attack_map = np.zeros((8, 8), dtype=np.int8)
    defense_map = np.zeros((8, 8), dtype=np.int8)

    for square in chess.SQUARES:
        attackers = board.attackers(chess.WHITE, square) | board.attackers(chess.BLACK, square)
        defenders = board.attackers(board.turn, square)
        
        row, col = divmod(square, 8)
        attack_map[row, col] = len(attackers)
        defense_map[row, col] = len(defenders)

    return attack_map, defense_map


def encode_move_relations(board, move):
    """
    Encodes move relationships by tracking board control changes.
    """
    from_square = move.from_square
    to_square = move.to_square
    piece = board.piece_at(from_square)

    # Before Move
    attack_before, defense_before = get_attack_defense_maps(board)

    # Apply Move
    board.push(move)
    attack_after, defense_after = get_attack_defense_maps(board)

    # Compute Changes
    changes = []
    for r in range(8):
        for c in range(8):
            if attack_after[r, c] != attack_before[r, c]:
                changes.append({"square": (r, c), "change": "control"})
            if defense_after[r, c] != defense_before[r, c]:
                changes.append({"square": (r, c), "change": "influence"})

    # Undo Move
    board.pop()

    return changes


def encode_move(board, move):
    """
    Converts a chess move into an enriched representation including attack/defense maps and positional motifs.
    """
    piece = board.piece_at(move.from_square)
    piece_symbol = piece.symbol().upper() if piece else "?"

    move_relations = encode_move_relations(board, move)

    attack_map, defense_map = get_attack_defense_maps(board)

    # Determine Game Phase
    num_pieces = len(board.piece_map())
    if num_pieces > 20:
        game_phase = "opening"
    elif num_pieces > 10:
        game_phase = "middlegame"
    else:
        game_phase = "endgame"

    return {
        "move": move.uci(),
        "from": divmod(move.from_square, 8),
        "to": divmod(move.to_square, 8),
        "piece_type": piece_symbol,
        "move_type": "capture" if board.is_capture(move) else "push",
        "piece_relations": move_relations,
        "attack_map": attack_map.tolist(),
        "defense_map": defense_map.tolist(),
        "game_phase": game_phase
    }

def normalize_board(board):
    """
    Converts a board position to a **White-perspective FEN**.
    If Black is to move, flips the board & converts to White's turn.
    Used for recall and storage to ensure **color resilience**.
    """
    if board.turn == chess.WHITE:
        return board  # No transformation needed

    # **Flip the board for White perspective**
    board_flipped = board.mirror()  # Flip ranks & pieces
    board_flipped.turn = chess.WHITE  # Ensure it's White's turn

    return board_flipped


import json
import os
import h5py
import chess
import chess.engine
import numpy as np
import tensorflow as tf
import random
import requests

HDF5_FILE = "training_data/training_data.hdf5"
VOCAB_FILE = "move_vocab.json"
MAX_MOVE_HISTORY = 50  # Fixed-length move history
STOCKFISH_API = "https://stockfish.online/api/s/v2.php"

STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"


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


# Build Move Vocabulary from HDF5
def build_move_vocab(hdf5_file=HDF5_FILE):
    """
    Builds a move vocabulary from an HDF5 dataset instead of JSON.
    """
    moves = set()

    with h5py.File(hdf5_file, "r") as hf:
        move_histories = hf["move_histories"][:]  

    for move_seq in move_histories:
        for move in move_seq:
            move_str = move.decode("ascii")  # Convert bytes to string
            if move_str.strip() and move_str != "000000":  # Ignore padding
                moves.add(move_str)

    move_vocab = {move: idx+1 for idx, move in enumerate(sorted(moves))}
    move_vocab["[PAD]"] = 0  # Padding token

    with open(VOCAB_FILE, "w") as f:
        json.dump(move_vocab, f, indent=4)

    return move_vocab


# Convert Move Sequences to Indexed List
def move_to_index(move_history, move_vocab, max_sequence_length=50):
    """
    Converts a sequence of PFFTTU moves into indexed representation with padding.
    """
    if isinstance(move_history, str):  
        move_history = [move_history]  # Convert single move to a list
    
    # Ensure each move is a string (handles NumPy scalars)
    indexed_moves = [move_vocab.get(str(move), 0) for move in move_history]

    if len(indexed_moves) < max_sequence_length:
        indexed_moves = [0] * (max_sequence_length - len(indexed_moves)) + indexed_moves
    else:
        indexed_moves = indexed_moves[-max_sequence_length:]

    return np.array(indexed_moves, dtype=np.int32)  


### 4️⃣ Load Vocabulary from File
def load_move_vocab():
    """
    Loads move vocabulary from JSON file.
    """
    if not os.path.exists(VOCAB_FILE):
        print("🚨 Move vocabulary file not found. Generating new vocab...")
        return build_move_vocab()

    with open(VOCAB_FILE, "r") as f:
        return json.load(f)



def index_to_move(move_index, board):
    """
    Converts an integer move index back into a valid chess move.

    Parameters:
        move_index (int): The index of the move (0-4095, assuming max 64x64 moves).
        board (chess.Board): The current chess board.

    Returns:
        chess.Move: The corresponding move.
    """
    legal_moves = list(board.legal_moves)  # Get all legal moves

    if move_index < len(legal_moves):  # Ensure the index is valid
        return legal_moves[move_index]
    
    # Fallback: If move index is invalid, choose a random legal move
    return random.choice(legal_moves)


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

    Returns:
        float: Evaluation score in pawns.
               - Positive = White is better
               - Negative = Black is better
               - ±1000 for checkmate situations
               - 0 if Stockfish fails
    """
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            eval_score = info["score"].relative  # Get relative evaluation
            
            if eval_score.is_mate():
                return 1000 if eval_score.mate() > 0 else -1000  # Checkmate detected
            
            return eval_score.score() / 100  # Convert centipawn to pawns
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

import json
import os
import h5py
import numpy as np
import tensorflow as tf

HDF5_FILE = "training_data/training_data.hdf5"
VOCAB_FILE = "move_vocab.json"
MAX_MOVE_HISTORY = 50  # Fixed-length move history


### 1️⃣ Convert FEN to Tensor Representation
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


### 2️⃣ Build Move Vocabulary from HDF5
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


### 3️⃣ Convert Move Sequences to Indexed List
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


### 5️⃣ Convert Indexed Moves Back to String
def index_to_move(indexed_moves, move_vocab):
    """
    Converts indexed moves back to human-readable PFFTTU format.
    """
    reverse_vocab = {idx: move for move, idx in move_vocab.items()}
    return [reverse_vocab.get(idx, "[UNK]") for idx in indexed_moves]

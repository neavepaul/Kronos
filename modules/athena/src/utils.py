import json
import os
import h5py
import json
import numpy as np
import tensorflow as tf

HDF5_FILE = "training_data/training_data.hdf5"

def fen_to_tensor(fen):
    """
    Converts a FEN string into a 8x8x20 tensor representation.
    """
    piece_mapping = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
    }

    # Initialize tensor with 12 channels for piece placement
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

    # Convert external information
    external_channels = np.zeros((8, 8, 8), dtype=np.float32)

    # Active Color Channel (8x8x1) - Fill with 1 for White to move, 0 for Black
    external_channels[:, :, 0] = 1.0 if active_color == "w" else 0.0

    # Castling Rights (8x8x4) - 1 if available, 0 otherwise
    castling_flags = ["K", "Q", "k", "q"]
    for j, flag in enumerate(castling_flags):
        if flag in castling_rights:
            external_channels[:, :, j + 1] = 1.0

    # En Passant Square (8x8x1)
    if en_passant != "-":
        col = ord(en_passant[0]) - ord("a")
        row = 8 - int(en_passant[1])  # Convert rank to row index
        external_channels[row, col, 5] = 1.0  # Mark en passant square

    # Halfmove Clock (8x8x1) - Normalize by 50
    external_channels[:, :, 6] = float(halfmove) / 50.0

    # Fullmove Number (8x8x1) - Normalize by 500 (adjust as needed)
    external_channels[:, :, 7] = float(fullmove) / 500.0

    # Combine tensors
    full_tensor = np.concatenate((tensor, external_channels), axis=-1)  # Shape: (8,8,20)

    return full_tensor



def build_move_vocab(hdf5_file=HDF5_FILE):
    """
    Builds a move vocabulary from an HDF5 dataset instead of JSON.
    """
    moves = set()

    # Open HDF5 file and read move indices
    with h5py.File(hdf5_file, "r") as hf:
        move_indices = hf["move_indices"][:]  # Load all move indices

    # Convert move indices to UCI notation (Assuming index → move mapping exists)
    for move_idx in move_indices:
        # If move_idx is valid (not padding), add it to the move set
        if move_idx > 0:
            moves.add(str(move_idx))  # Store as a string (same format as before)

    # Assign each unique move an index
    move_vocab = {move: idx+1 for idx, move in enumerate(sorted(moves))}

    # Save vocabulary to JSON
    with open("move_vocab.json", "w") as f:
        json.dump(move_vocab, f)

    return move_vocab


# Convert move sequence to indexed list
def move_to_index(move_history, move_vocab, max_sequence_length=50):
    """
    Converts a sequence of moves into indexed representation with padding.
    """
    indexed_moves = [move_vocab.get(move, 0) for move in move_history]  # 0 for unknown moves

    # Ensure fixed sequence length
    if len(indexed_moves) < max_sequence_length:
        # Pad with zeros at the start
        indexed_moves = [0] * (max_sequence_length - len(indexed_moves)) + indexed_moves
    else:
        # Truncate to last `max_sequence_length` moves
        indexed_moves = indexed_moves[-max_sequence_length:]

    return np.array(indexed_moves, dtype=np.int32)  # Ensure NumPy array



class ChessDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, hdf5_path, batch_size=32, shuffle=True):
        self.h5 = h5py.File(hdf5_path, 'r')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.h5['targets']))
        
    def __getitem__(self, idx):
        batch_idx = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        fens = self.h5['fens'][batch_idx]
        histories = self.h5['move_history'][batch_idx]
        masks = self.h5['legal_masks'][batch_idx]
        targets = self.h5['targets'][batch_idx]
        
        # Convert to one-hot
        target_moves = np.zeros((self.batch_size, 64, 64))
        for i, (frm, to) in enumerate(targets):
            target_moves[i, frm, to] = 1
            
        return [fens, histories, masks], [target_moves, masks]
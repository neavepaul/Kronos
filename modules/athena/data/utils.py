import numpy as np
import json
import os



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


# Build move vocabulary from multiple JSONs
def build_move_vocab(json_dir):
    moves = set()
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            with open(os.path.join(json_dir, json_file), "r") as f:
                games = json.load(f)
                for game in games:
                    for position in game:
                        moves.add(position["move"])
    
    move_vocab = {move: idx+1 for idx, move in enumerate(sorted(moves))}
    
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

from multiprocessing import Pool, cpu_count, Process, Manager
import h5py
import numpy as np
import os
import gc
import chess
import chess.pgn
import chess.engine
from tqdm import tqdm
from utils import fen_to_tensor

# Global Constants
HDF5_FILE = "training_data/training_data.hdf5"
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
MAX_MOVE_HISTORY = 50  # Fixed-length move history

def uci_move_to_index(board, move):
    """
    Converts a UCI move string to a structured PFFTTU token.
    Fixes AssertionError by detecting the piece BEFORE pushing the move.
    """
    piece_mapping = {
        "P": "P", "N": "N", "B": "B", "R": "R", "Q": "Q", "K": "K",
        "p": "p", "n": "n", "b": "b", "r": "r", "q": "q", "k": "k"
    }

    # Extract the "from" and "to" squares from the UCI move
    from_square = chess.parse_square(move[:2])
    to_square = chess.parse_square(move[2:4])

    # Extract piece BEFORE making the move
    piece = board.piece_at(from_square)

    # Ensure we don't call .symbol() on NoneType
    piece_symbol = piece_mapping.get(piece.symbol(), "?") if piece else "?"

    # Handle promotion (e.g., e7e8q)
    promo_piece_map = {"q": "Q", "r": "R", "b": "B", "n": "N"}
    promo_piece = promo_piece_map.get(move[4], "-") if len(move) == 5 else "-"

    return f"{piece_symbol}{from_square:02}{to_square:02}{promo_piece}"

def process_game(args):
    """
    Processes a single game by extracting FEN tensors, move histories, legal move masks, evaluation scores, and next moves.
    """
    moves, game_id, engine_path, queue = args
    board = chess.Board()
    move_history = []
    game_data = []

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        outcome = board.outcome()
        winner = 1 if outcome and outcome.winner == chess.WHITE else (-1 if outcome and outcome.winner == chess.BLACK else 0)

        for move in moves:
            fen_tensor = fen_to_tensor(board.fen())

            # Compute legal move mask (64x64)
            legal_moves_mask = np.zeros((64, 64), dtype=np.int8)
            for legal_move in board.legal_moves:
                legal_moves_mask[legal_move.from_square, legal_move.to_square] = 1

            turn_indicator = 1.0 if board.turn == chess.WHITE else 0.0


            # Compute evaluation score from Stockfish
            try:
                eval_result = engine.analyse(board, chess.engine.Limit(depth=10))
                eval_score = eval_result["score"].relative.score(mate_score=10000)
            except Exception:
                eval_score = 0  

            # Extract the move encoding BEFORE pushing the move
            move_encoded = uci_move_to_index(board, move)

            # Apply the move after extracting it
            board.push(chess.Move.from_uci(move))
            move_history.append(move_encoded)

            move_history_padded = ["000000"] * MAX_MOVE_HISTORY  
            move_indices = move_history[-MAX_MOVE_HISTORY:]
            move_history_padded[-len(move_indices):] = move_indices  

            game_data.append({
                "fen_tensor": fen_tensor,
                "move_history": move_history_padded,
                "legal_moves_mask": legal_moves_mask,
                "eval_score": eval_score,
                "next_move": move_encoded,
                "turn_indicator": turn_indicator
            })

    queue.put(game_data)
    gc.collect()
    return True

def hdf5_writer(queue):
    """ Writes processed game data to an HDF5 file. """
    with h5py.File(HDF5_FILE, "w", libver="latest") as hf:
        hf.swmr_mode = True
        hf.create_dataset("fens", shape=(0, 8, 8, 20), maxshape=(None, 8, 8, 20), dtype=np.float32)
        hf.create_dataset("move_histories", shape=(0, MAX_MOVE_HISTORY), maxshape=(None, MAX_MOVE_HISTORY), dtype="S6")
        hf.create_dataset("legal_moves_mask", shape=(0, 64, 64), maxshape=(None, 64, 64), dtype=np.int8)
        hf.create_dataset("eval_score", shape=(0, 1), maxshape=(None, 1), dtype=np.float32)
        hf.create_dataset("turn_indicator", shape=(0, 1), maxshape=(None, 1), dtype=np.float32)
        hf.create_dataset("next_move", shape=(0,), maxshape=(None,), dtype="S6")

        while True:
            game_data = queue.get()
            if game_data is None:
                break
            if not game_data:
                continue  # Skip empty data

            # Extract arrays from the game data
            fens = np.array([data["fen_tensor"] for data in game_data], dtype=np.float32)
            turn_indicators = np.array([[data["turn_indicator"]] for data in game_data], dtype=np.float32)
            eval_scores = np.array([[data["eval_score"]] for data in game_data], dtype=np.float32)
            legal_moves = np.array([data["legal_moves_mask"] for data in game_data], dtype=np.int8)

            # Handle move history encoding properly
            move_histories = [
                [m.encode("ascii") for m in data["move_history"][-MAX_MOVE_HISTORY:]]
                if len(data["move_history"]) >= MAX_MOVE_HISTORY
                else [b"000000"] * (MAX_MOVE_HISTORY - len(data["move_history"])) + 
                     [m.encode("ascii") for m in data["move_history"]]
                for data in game_data
            ]
            move_histories = np.array(move_histories, dtype="S6")  # Convert to NumPy array

            # Handle next moves properly
            next_moves = np.array([data["next_move"].encode("ascii") for data in game_data], dtype="S6")

            # Resize and write to HDF5
            hf["fens"].resize(hf["fens"].shape[0] + fens.shape[0], axis=0)
            hf["fens"][-fens.shape[0]:] = fens

            hf["turn_indicator"].resize(hf["turn_indicator"].shape[0] + turn_indicators.shape[0], axis=0)
            hf["turn_indicator"][-turn_indicators.shape[0]:] = turn_indicators

            hf["eval_score"].resize(hf["eval_score"].shape[0] + eval_scores.shape[0], axis=0)
            hf["eval_score"][-eval_scores.shape[0]:] = eval_scores

            hf["legal_moves_mask"].resize(hf["legal_moves_mask"].shape[0] + legal_moves.shape[0], axis=0)
            hf["legal_moves_mask"][-legal_moves.shape[0]:] = legal_moves

            hf["move_histories"].resize(hf["move_histories"].shape[0] + move_histories.shape[0], axis=0)
            hf["move_histories"][-move_histories.shape[0]:] = move_histories

            hf["next_move"].resize(hf["next_move"].shape[0] + next_moves.shape[0], axis=0)
            hf["next_move"][-next_moves.shape[0]:] = next_moves


def extract_games_from_pgn(pgn_file):
    """Extracts all games from a PGN file."""
    games = []
    with open(pgn_file) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            moves = [move.uci() for move in game.mainline_moves()]
            games.append(moves)
    return games


def process_pgn_in_parallel(pgn_file, engine_path, num_workers=cpu_count()):
    """Processes all games inside a PGN file in parallel while displaying progress bars."""
    print(f"Processing {pgn_file}...")

    games = extract_games_from_pgn(pgn_file)

    manager = Manager()
    queue = manager.Queue(maxsize=10)

    writer_process = Process(target=hdf5_writer, args=(queue,))
    writer_process.start()

    jobs = [(moves, i, engine_path, queue) for i, moves in enumerate(games)]  # Pass all required arguments

    with tqdm(total=len(games), desc=f"Processing {pgn_file}") as pbar:
        with Pool(num_workers) as pool:
            for _ in pool.imap_unordered(process_game, jobs):  # Now properly formatted!
                pbar.update(1)

    queue.put(None)
    writer_process.join()
    print(f"Completed processing {pgn_file}")


def main():
    """Main function to process all PGN files."""
    if os.path.exists(HDF5_FILE):
        os.remove(HDF5_FILE)
        print(f"Deleted file: {HDF5_FILE}")

    engine_path = STOCKFISH_PATH
    pgn_dir = "pgn_files"
    pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith(".pgn")]

    for pgn_file in pgn_files:
        process_pgn_in_parallel(os.path.join(pgn_dir, pgn_file), engine_path)


if __name__ == "__main__":
    main()

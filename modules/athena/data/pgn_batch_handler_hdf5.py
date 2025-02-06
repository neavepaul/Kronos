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
    Converts a UCI move string to a structured 6-character zero-padded format:
    PFFTTT (Piece, From-Square, To-Square, Promotion)
    """
    piece_mapping = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
    }

    from_square = chess.parse_square(move[:2])  # Convert "e2" -> 12
    to_square = chess.parse_square(move[2:4])   # Convert "e4" -> 28
    piece = board.piece_at(from_square)

    # Use FEN-style indexing for piece types
    piece_type = piece_mapping[piece.symbol()] if piece else 0  

    # Handle promotion (e.g., e7e8q)
    promo_piece_map = {"q": 4, "r": 3, "b": 2, "n": 1}  # Align promotion pieces with `fen_to_tensor`
    promo_piece = promo_piece_map.get(move[4], 0) if len(move) == 5 else 0

    return f"{piece_type}{from_square:02}{to_square:02}{promo_piece}".zfill(6)


def process_game(args):
    """
    Processes a single game by extracting FEN tensors, move histories, legal move masks, evaluation scores, and next moves.
    """
    moves, game_id, engine_path, queue = args  # ✅ Unpacking arguments properly
    board = chess.Board()
    move_history = []
    game_data = []

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        outcome = board.outcome()
        winner = 1 if outcome and outcome.winner == chess.WHITE else (-1 if outcome and outcome.winner == chess.BLACK else 0)

        for i, move in enumerate(moves):
            fen_tensor = fen_to_tensor(board.fen())

            # Compute legal move mask (64x64)
            legal_moves_mask = np.zeros((64, 64), dtype=np.int8)
            for legal_move in board.legal_moves:
                legal_moves_mask[legal_move.from_square, legal_move.to_square] = 1

            # Compute evaluation score from Stockfish
            try:
                eval_result = engine.analyse(board, chess.engine.Limit(depth=10))
                eval_score = eval_result["score"].relative.score(mate_score=10000)
            except Exception:
                eval_score = 0  

            if winner == -1:
                fen_tensor = np.flip(fen_tensor, axis=(0, 1))
                legal_moves_mask = np.flip(legal_moves_mask, axis=(0, 1))
                eval_score *= -1  

            move_history_padded = ["000000"] * MAX_MOVE_HISTORY  
            move_indices = [uci_move_to_index(board, m) for m in move_history[-MAX_MOVE_HISTORY:]]
            move_history_padded[-len(move_indices):] = move_indices  

            next_move_encoded = uci_move_to_index(board, move)

            game_data.append({
                "fen_tensor": fen_tensor,
                "move_history": move_history_padded,
                "legal_moves_mask": legal_moves_mask,
                "eval_score": eval_score,
                "next_move": next_move_encoded
            })

            move_history.append(move)
            board.push(chess.Move.from_uci(move))

    queue.put(game_data)
    gc.collect()
    return True


def hdf5_writer(queue):
    """
    Writes processed game data to an HDF5 file.
    """
    with h5py.File(HDF5_FILE, "w", libver="latest") as hf:
        hf.swmr_mode = True

        hf.create_dataset("fens", shape=(0, 8, 8, 20), maxshape=(None, 8, 8, 20), dtype=np.float32)
        hf.create_dataset("move_histories", shape=(0, MAX_MOVE_HISTORY), maxshape=(None, MAX_MOVE_HISTORY), dtype="S6")
        hf.create_dataset("legal_moves_mask", shape=(0, 64, 64), maxshape=(None, 64, 64), dtype=np.int8)
        hf.create_dataset("eval_score", shape=(0, 1), maxshape=(None, 1), dtype=np.float32)
        hf.create_dataset("next_move", shape=(0,), maxshape=(None,), dtype="S6")

        while True:
            game_data = queue.get()
            if game_data is None:
                break  # Stop if None is received (signal to terminate)

            if not game_data:
                continue  # Skip empty data

            fens = np.array([data["fen_tensor"] for data in game_data], dtype=np.float32)

            # ✅ **Ensure move history is always MAX_MOVE_HISTORY length**
            move_histories = np.array([
                np.array([m.encode("ascii") for m in data["move_history"]], dtype="S6")
                if len(data["move_history"]) == MAX_MOVE_HISTORY
                else np.array([b"000000"] * (MAX_MOVE_HISTORY - len(data["move_history"])) + 
                              [m.encode("ascii") for m in data["move_history"][-MAX_MOVE_HISTORY:]], dtype="S6")
                for data in game_data
            ])

            next_moves = np.array([data["next_move"].encode("ascii") for data in game_data], dtype="S6")
            legal_moves = np.array([data["legal_moves_mask"] for data in game_data], dtype=np.int8)
            eval_scores = np.array([[data["eval_score"]] for data in game_data], dtype=np.float32)

            # ✅ **Ensure Fixed Shapes Before Writing**
            assert move_histories.shape[1] == MAX_MOVE_HISTORY, f"Move history shape mismatch: {move_histories.shape}"

            # Resize and write to HDF5
            hf["fens"].resize(hf["fens"].shape[0] + fens.shape[0], axis=0)
            hf["fens"][-fens.shape[0]:] = fens

            hf["move_histories"].resize(hf["move_histories"].shape[0] + move_histories.shape[0], axis=0)
            hf["move_histories"][-move_histories.shape[0]:] = move_histories

            hf["legal_moves_mask"].resize(hf["legal_moves_mask"].shape[0] + legal_moves.shape[0], axis=0)
            hf["legal_moves_mask"][-legal_moves.shape[0]:] = legal_moves

            hf["eval_score"].resize(hf["eval_score"].shape[0] + eval_scores.shape[0], axis=0)
            hf["eval_score"][-eval_scores.shape[0]:] = eval_scores

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

    jobs = [(moves, i, engine_path, queue) for i, moves in enumerate(games)]  # ✅ Pass all required arguments

    with tqdm(total=len(games), desc=f"Processing {pgn_file}") as pbar:
        with Pool(num_workers) as pool:
            for _ in pool.imap_unordered(process_game, jobs):  # ✅ Now properly formatted!
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

from multiprocessing import Pool, cpu_count, Process, Manager
import h5py
import numpy as np
import os
import gc
import chess
import chess.pgn
import chess.engine
from tqdm import tqdm
from utils import fen_to_tensor, build_move_vocab, move_to_index

HDF5_FILE = "training_data/training_data.hdf5"

def process_game(moves, game_id, engine_path, move_vocab, queue, max_sequence_length=50):
    board = chess.Board()
    move_history = []
    game_data = []

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for move in moves:
            fen_tensor = fen_to_tensor(board.fen())
            evaluation = engine.analyse(board, chess.engine.Limit(depth=6))
            score = evaluation["score"].relative.score(mate_score=10000)

            move_index = move_vocab.get(move, 0)  # Default to 0 for unknown moves
            move_history_indices = move_to_index(move_history, move_vocab, max_sequence_length)

            game_data.append({
                "fen_tensor": fen_tensor,
                "move_history": move_history_indices,
                "evaluation": score / 100.0,
                "move_index": move_index
            })

            move_history.append(move)
            board.push(chess.Move.from_uci(move))

    # Send processed data to the writer process via the queue
    queue.put(game_data)

    gc.collect()
    return True  # or any marker to indicate the job is done

def hdf5_writer(queue):
    with h5py.File(HDF5_FILE, "w", libver="latest") as hf:
        hf.swmr_mode = True  # Enable SWMR mode

        # Initialize HDF5 datasets with unlimited size
        hf.create_dataset("fens", shape=(0, 8, 8, 20),
                          maxshape=(None, 8, 8, 20), dtype=np.float32)
        hf.create_dataset("move_histories", shape=(0, 50),
                          maxshape=(None, 50), dtype=np.int32)
        hf.create_dataset("evaluations", shape=(0,),
                          maxshape=(None,), dtype=np.float32)
        hf.create_dataset("move_indices", shape=(0,),
                          maxshape=(None,), dtype=np.int32)

        while True:
            game_data = queue.get()
            if game_data is None:
                break  # Stop if None is received (signal to terminate)
            
            # >>> Skip writing if there's no data <<<
            if not game_data:  
                continue

            fens = np.array([data["fen_tensor"] for data in game_data], dtype=np.float32)
            move_histories = np.array([data["move_history"] for data in game_data], dtype=np.int32)
            evaluations = np.array([data["evaluation"] for data in game_data], dtype=np.float32)
            move_indices = np.array([data["move_index"] for data in game_data], dtype=np.int32)

            # Resize and write to HDF5
            hf["fens"].resize(hf["fens"].shape[0] + fens.shape[0], axis=0)
            hf["fens"][-fens.shape[0]:] = fens

            hf["move_histories"].resize(hf["move_histories"].shape[0] + move_histories.shape[0], axis=0)
            hf["move_histories"][-move_histories.shape[0]:] = move_histories

            hf["evaluations"].resize(hf["evaluations"].shape[0] + evaluations.shape[0], axis=0)
            hf["evaluations"][-evaluations.shape[0]:] = evaluations

            hf["move_indices"].resize(hf["move_indices"].shape[0] + move_indices.shape[0], axis=0)
            hf["move_indices"][-move_indices.shape[0]:] = move_indices

def extract_games_from_pgn(pgn_file):
    games = []
    with open(pgn_file) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            moves = [move.uci() for move in game.mainline_moves()]
            games.append(moves)
    return games

def process_game_wrapper(args):
    """
    Simple wrapper to unpack arguments and call process_game.
    This lets us use pool.imap_unordered(...), which plays nicely with tqdm.
    """
    return process_game(*args)

def process_pgn_in_parallel(pgn_file, engine_path, move_vocab, num_workers=cpu_count()):
    print(f"Processing {pgn_file}...")

    # Extract all games from this PGN
    games = extract_games_from_pgn(pgn_file)

    manager = Manager()
    queue = manager.Queue(maxsize=10)

    writer_process = Process(target=hdf5_writer, args=(queue,))
    writer_process.start()

    # Create the job list for pool.imap_unordered
    jobs = [(moves, i, engine_path, move_vocab, queue) for i, moves in enumerate(games)]

    # Use a Pool to process games in parallel, displaying a progress bar.
    with Pool(num_workers) as pool:
        # total=len(jobs) ensures the tqdm bar runs to the total number of games
        for _ in tqdm(pool.imap_unordered(process_game_wrapper, jobs),
                      total=len(jobs),
                      desc=f"Games in {os.path.basename(pgn_file)}"):
            pass  # We only care about the progress, not the return values.

    # Signal writer process to terminate
    queue.put(None)
    writer_process.join()
    print(f"Completed processing {pgn_file}")

def main():
    if os.path.exists(HDF5_FILE):
        os.remove(HDF5_FILE)
        print(f"Deleted file: {HDF5_FILE}")
    else:
        print(f"File does not exist: {HDF5_FILE}")
    engine_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
    pgn_dir = "pgn_files"
    move_vocab = {}  # Load or build move vocab if available

    pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith(".pgn")]

    # Show a progress bar for the files themselves
    for pgn_file in tqdm(pgn_files, desc="PGN Files"):
        pgn_path = os.path.join(pgn_dir, pgn_file)
        process_pgn_in_parallel(pgn_path, engine_path, move_vocab)

if __name__ == "__main__":
    main()

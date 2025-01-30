import chess
import chess.pgn
import chess.engine
import json
import os
import time
import gc  # Garbage collector
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Process, Value, Lock

def extract_games_from_pgn(pgn_file):
    """
    Reads a PGN file and extracts complete games as a list of UCI moves.
    """
    games = []
    with open(pgn_file) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            
            # Convert to a primitive structure: list of UCI moves
            moves = [move.uci() for move in game.mainline_moves()]
            games.append(moves)

    return games

def process_game(moves, game_id, engine_path):
    """
    Processes a full game (list of UCI moves) and evaluates each position.
    """
    board = chess.Board()
    game_data = []

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for move in moves:
            fen = board.fen()
            
            # Evaluate position
            evaluation = engine.analyse(board, chess.engine.Limit(depth=6))
            score = evaluation["score"].relative.score(mate_score=10000)

            game_data.append({
                "game_id": game_id,
                "fen": fen,
                "move": move,
                "evaluation": score / 100.0
            })

            board.push(chess.Move.from_uci(move))

    # **Write the full processed game to disk at once**
    with lock:
        with open(output_file_path, "a") as f:
            json.dump(game_data, f, indent=4)
            f.write(",\n")  # Add comma to separate games

        progress_counter.value += 1  # Update progress bar

    gc.collect()  # Prevent memory leaks
    return True  # Indicating successful processing

def init_globals(shared_progress_counter, shared_lock, output_file):
    """
    Initialize global variables for worker processes.
    """
    global progress_counter, lock, output_file_path
    progress_counter = shared_progress_counter
    lock = shared_lock
    output_file_path = output_file

def process_chunk(args):
    """
    Process a single game (list of UCI moves) and evaluate each position.
    """
    moves, game_id, engine_path = args  # Unpack arguments
    return process_game(moves, game_id, engine_path)

def tqdm_updater(progress_counter, total_games, lock):
    """
    Updates the tqdm progress bar in the main process.
    """
    with tqdm(total=total_games, desc="Processing Games", position=0, leave=True) as pbar:
        while True:
            with lock:
                pbar.n = progress_counter.value
                pbar.refresh()
            
            if progress_counter.value >= total_games:
                break  # Stop when all games are processed

            time.sleep(0.5)  # Small delay to prevent excessive updates

def process_pgn_in_parallel(pgn_file, engine_path, output_file, num_workers=cpu_count()):
    """
    Processes a PGN file in parallel while ensuring full games remain intact.
    """
    print(f"Processing {pgn_file}...")

    # Step 1: Extract full games as lists of UCI moves (avoids passing unpicklable objects)
    games = extract_games_from_pgn(pgn_file)

    # Step 2: Create shared progress counter and lock
    shared_progress_counter = Value("i", 0)  # Shared integer for tracking progress
    shared_lock = Lock()  # Lock for safe updates

    # Step 3: Prepare JSON file
    with open(output_file, "w") as f:
        f.write("[\n")  # Start JSON array

    # Step 4: Start the TQDM updater as a separate process (pass shared objects)
    tqdm_process = Process(target=tqdm_updater, args=(shared_progress_counter, len(games), shared_lock))
    tqdm_process.start()

    # Step 5: Process games using `imap_unordered()` to stream results back
    with Pool(num_workers, initializer=init_globals, initargs=(shared_progress_counter, shared_lock, output_file)) as pool:
        for _ in pool.imap_unordered(process_chunk, [(moves, i, engine_path) for i, moves in enumerate(games)]):
            pass  # We don't store results in memory

    tqdm_process.join()  # Ensure progress bar updates are completed

    # Step 6: Finalize JSON file properly
    with open(output_file, "a") as f:
        f.write("\n]")  # End JSON array

    print(f"Training data saved to {output_file}")

def main():
    engine_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
    training_data_dir = "training_data"
    
    pgn_file = "pgn_files/Tal.pgn"

    if not os.path.exists(training_data_dir):
        os.makedirs(training_data_dir)

    output_file = os.path.join(training_data_dir, f"{os.path.splitext(os.path.basename(pgn_file))[0]}_training_data.json")
    process_pgn_in_parallel(pgn_file, engine_path, output_file)

if __name__ == "__main__":
    main()

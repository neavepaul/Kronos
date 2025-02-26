import numpy as np
import h5py
import json
import chess
import chess.polyglot  # For hashing board positions

from modules.athena.src.utils import normalize_board

MEMORY_FILE = "game_memory.hdf5"
MAX_GAMES_STORED = 100000

class GameMemory:
    """Stores self-play games for pattern recognition & experience replay."""

    def __init__(self):
        self.memory = []

    def add_game(self, move_sequence, final_eval, final_fen):
        """
        Stores a sequence of moves as a training example.
        Now stores board state in a **normalized White-to-move** format.
        """
        if len(self.memory) >= MAX_GAMES_STORED:
            self.memory.pop(0)  # Remove oldest game

        # **Normalize board FEN before storing**
        board = chess.Board(fen=final_fen)
        normalized_fen = normalize_board(board).fen()

        # Store moves with the **normalized** board position
        self.memory.append({"moves": move_sequence, "eval": final_eval, "fen": normalized_fen})


    def find_similar_position(self, board, eval_threshold=50):
        """
        Finds a past game with a similar board position using hashing.
        
        **Now resilient to color changes!**
        Stores and retrieves positions in a White-to-move perspective.
        Normalizes board state before storing & recalling.
        
        Returns:
            - Best-matching game with an eval > `eval_threshold`
            - None if no strong recallable move exists
        """
        # **Normalize Board State Before Searching**
        board_normalized = normalize_board(board)
        board_hash = chess.polyglot.zobrist_hash(board_normalized)

        best_match = None
        best_eval = -float('inf')

        for game in self.memory:
            try:
                # **Ensure data integrity**
                if "fen" not in game or "eval" not in game:
                    print(f"⚠️ Skipping game due to missing 'fen' or 'eval' field: {game}")
                    continue

                # **Normalize Past Game Position**
                past_board = chess.Board(fen=game["fen"])
                past_board_normalized = normalize_board(past_board)
                game_hash = chess.polyglot.zobrist_hash(past_board_normalized)

                # **Match only if eval is significantly positive**
                if game_hash == board_hash and game["eval"] > eval_threshold and game["eval"] > best_eval:
                    best_match = game
                    best_eval = game["eval"]

            except Exception as e:
                print(f"⚠️ Error processing stored game: {e}")
                continue  # Skip corrupted games

        return best_match

    def sample_games(self, num_samples):
        """Retrieves past game patterns for training."""
        num_samples = min(num_samples, len(self.memory))
        return np.random.choice(self.memory, num_samples, replace=False)

    def save(self):
        """Saves self-play memory to an HDF5 file."""
        with h5py.File(MEMORY_FILE, "w") as f:
            games = [json.dumps(game) for game in self.memory]
            f.create_dataset("games", data=np.array(games, dtype="S200"))

    def load(self):
        """Loads stored self-play games from file."""
        try:
            with h5py.File(MEMORY_FILE, "r") as f:
                self.memory = [json.loads(game.decode("utf-8")) for game in f["games"][:]]
        except FileNotFoundError:
            print("No previous game memory found.")

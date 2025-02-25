import numpy as np
import h5py
import chess
import chess.polyglot  # For hashing board positions

MEMORY_FILE = "game_memory.hdf5"
MAX_GAMES_STORED = 100000

class GameMemory:
    """Stores self-play games for pattern recognition & experience replay."""

    def __init__(self):
        self.memory = []

    def add_game(self, move_sequence, final_eval):
        """Stores a sequence of moves as a training example."""
        if len(self.memory) >= MAX_GAMES_STORED:
            self.memory.pop(0)  # Remove the oldest game

        # Save moves along with final Stockfish eval to prioritize strong games
        self.memory.append({"moves": move_sequence, "eval": final_eval})

    def find_similar_position(self, board):
        """Finds a past game with a similar board position using hashing."""
        board_hash = chess.polyglot.zobrist_hash(board)
        best_match = None
        best_eval = -float('inf')

        for game in self.memory:
            game_hash = chess.polyglot.zobrist_hash(chess.Board(fen=game["moves"][0]))
            if game_hash == board_hash and game["eval"] > best_eval:
                best_match = game
                best_eval = game["eval"]

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

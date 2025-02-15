import chess
import chess.syzygy
from pathlib import Path

class Hades:
    def __init__(self, syzygy_path):
        """Initialize Hades with Syzygy tablebases."""
        self.syzygy_path = Path(syzygy_path)
        self.tablebase = chess.syzygy.Tablebase()

        try:
            self.tablebase.add_directory(str(self.syzygy_path))
            print(f"Hades: Syzygy Tablebase Loaded from {self.syzygy_path}")
        except Exception as e:
            print(f"⚠️ Error Loading Syzygy Tablebases: {e}")

    def get_best_endgame_move(self, board):
        """Returns the best move if it's a tablebase position, otherwise None."""
        try:
            if len(board.piece_map()) <= 5:  # Syzygy 3-4-5 tablebases
                for move in board.legal_moves:
                    board.push(move)
                    if self.tablebase.probe_wdl(board) == 2:  # Win
                        board.pop()
                        print(f"🔥 Hades: Found Winning Move {move}")
                        return move
                    board.pop()
        except Exception as e:
            print(f"⚠️ Hades Error: {e}")

        return None  # No endgame move found

    def close(self):
        """Close tablebase to free resources."""
        self.tablebase.close()
        print("🔒 Hades: Tablebase Closed")

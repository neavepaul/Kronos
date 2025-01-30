from move_generator import generate_moves
from game_history import GameHistory
from utils import print_board, is_valid_fen
import chess

class Orchestrator:
    def __init__(self):
        """
        Initialize the orchestrator and its subsystems.
        """
        self.history = GameHistory()

    def make_move(self, fen, move):
        """
        Make a move, update the game history, and return the new position.
        :param fen: FEN string of the current position.
        :param move: Move in UCI format.
        :return: Updated FEN string after the move.
        """
        if not is_valid_fen(fen):
            raise ValueError("Invalid FEN string!")

        board = chess.Board(fen)
        if chess.Move.from_uci(move) in board.legal_moves:
            board.push(chess.Move.from_uci(move))
            new_fen = board.fen()

            # Add to history
            self.history.add_move(fen, move)
            return new_fen
        else:
            raise ValueError("Invalid move!")

    def get_game_history(self):
        """
        Retrieve the complete game history.
        """
        return self.history.get_history()


if __name__ == "__main__":
    # Example usage of Zeus
    orchestrator = Orchestrator()

    initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move = "e2e4"

    # Make a move and update history
    new_fen = orchestrator.make_move(initial_fen, move)
    print(f"New position: {new_fen}")

    # Print the game history
    history = orchestrator.get_game_history()
    for entry in history:
        print(f"Position: {entry['fen']} | Move: {entry['move']}")

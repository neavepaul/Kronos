from opening_book import OpeningBook
from move_history import MoveHistory
import chess

class Apollo:
    def __init__(self, book_path):
        """
        Initialize Apollo (Opening Module).
        :param book_path: Path to the Polyglot book file.
        """
        self.opening_book = OpeningBook(book_path)
        self.move_history = MoveHistory()

    def get_move(self, fen):
        """
        Get the best move for the given position during the opening phase.
        :param fen: FEN string of the current board position.
        :return: The best move as a UCI string, or None if no move is found.
        """
        board = chess.Board(fen)

        # Query the opening book
        move = self.opening_book.get_opening_move(board)
        if move:
            # Add the move to move history
            self.move_history.add_move(move)
            return move

        # No move found in the book
        return None

if __name__ == "__main__":
    # Path to the merged Polyglot book
    BOOK_PATH = "data/merged_book.bin"

    # Initialize Apollo
    apollo = Apollo(BOOK_PATH)

    # Example board position (FEN for starting position)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move = apollo.get_move(fen)

    if move:
        print(f"Best opening move: {move}")
    else:
        print("No move found in the opening book.")

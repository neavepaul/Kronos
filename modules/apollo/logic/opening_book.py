import chess
import chess.polyglot

class OpeningBook:
    def __init__(self, book_path):
        """
        Initialize the opening book.
        :param book_path: Path to the Polyglot book file.
        """
        self.book_path = book_path

    def get_opening_move(self, board):
        """
        Query the opening book for the best move in the given position.
        :param board: A chess.Board object representing the current position.
        :return: The best move as a UCI string, or None if not found in the book.
        """
        try:
            with chess.polyglot.open_reader(self.book_path) as reader:
                entry = reader.find(board)
                return entry.move.uci()
        except IndexError:
            # No moves found in the book for this position
            return None

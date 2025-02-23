import chess
import chess.polyglot

class Apollo:
    def __init__(self, book_path):
        """Initialize Apollo with the opening book."""
        self.book = chess.polyglot.open_reader(book_path)

    def get_opening_move(self, board):
        """Retrieves the best opening move from Apollo's book (highest weight)."""
        try:
            book_moves = list(self.book.find_all(board))
            if book_moves:
                # Sort moves by weight (descending) and select the best one
                best_move = max(book_moves, key=lambda move: move.weight).move
                print(f"📖 Apollo Opening Move: {best_move}")
                return best_move
        except StopIteration:
            pass

        print("❌ No book move found in Apollo.")
        return None

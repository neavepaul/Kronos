import chess
import chess.polyglot
import random

class Apollo:
    def __init__(self, book_path):
        """Initialize Apollo with the opening book."""
        self.book = chess.polyglot.open_reader(book_path)

    def get_opening_move(self, board):
        """Retrieves an opening move from Apollo's book, chosen randomly."""
        try:
            book_moves = list(self.book.find_all(board))
            if book_moves:
                chosen_move = random.choice(book_moves).move
                print(f"📖 Apollo Opening Move: {chosen_move}")
                return chosen_move
        except StopIteration:
            pass

        print("❌ No book move found in Apollo.")
        return None

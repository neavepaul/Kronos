import chess

def is_valid_fen(fen):
    """
    Check if the given FEN string is valid.
    """
    try:
        chess.Board(fen)
        return True
    except ValueError:
        return False

def print_board(fen):
    """
    Print a visual representation of the board from a FEN string.
    """
    board = chess.Board(fen)
    print(board)

if __name__ == "__main__":
    # Example usage
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    if is_valid_fen(fen):
        print_board(fen)
    else:
        print("Invalid FEN!")

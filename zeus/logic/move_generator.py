import chess

def generate_moves(fen):
    """
    Generate all legal moves for a given position in FEN format.
    """
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    return [move.uci() for move in moves]

if __name__ == "__main__":
    # Example FEN for initial position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    legal_moves = generate_moves(fen)
    print(f"Legal moves: {legal_moves}")


from zeus.logic.move_generator import generate_moves
from zeus.logic.utils import print_board, is_valid_fen

def test_move_generation():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    if is_valid_fen(fen):
        print("Initial position:")
        print_board(fen)
        legal_moves = generate_moves(fen)
        print(f"Legal moves: {legal_moves}")
        print(len(legal_moves))
    else:
        print("Invalid FEN string!")

if __name__ == "__main__":
    test_move_generation()

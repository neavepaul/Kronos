from evaluation import Athena
from utils import print_board

def test_athena():
    athena = Athena()

    # Test position (FEN for initial position)
    fen = "8/8/8/4p1K1/2k1P3/8/8/8 b - - 0 1"
    print("Testing position:")
    print_board(fen)

    # Evaluate position
    score = athena.evaluate_position(fen)
    print(f"Position score: {score}")

if __name__ == "__main__":
    test_athena()

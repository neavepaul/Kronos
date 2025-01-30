from orchestrator import Orchestrator
from utils import print_board

def test_zeus():
    orchestrator = Orchestrator()

    # Initial position
    initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print("Initial position:")
    print_board(initial_fen)

    # Make a move
    move = "e2e4"
    print(f"Making move: {move}")
    new_fen = orchestrator.make_move(initial_fen, move)
    print("Position after move:")
    print_board(new_fen)

    move = "e7e5"
    print(f"Making move: {move}")
    new_fen = orchestrator.make_move(new_fen, move)
    print("Position after move:")
    print_board(new_fen)

    # Get game history
    history = orchestrator.get_game_history()
    print("\nGame History:")
    for entry in history:
        print(f"Position: {entry['fen']} | Move: {entry['move']}")

if __name__ == "__main__":
    test_zeus()

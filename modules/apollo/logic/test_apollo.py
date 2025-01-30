import chess
import chess.engine
from apollo import Apollo

# Path to the Polyglot book
BOOK_PATH = "data/merged_book.bin"

def play_against_engine():
    """
    Test Apollo by playing it against a chess engine.
    """
    # Initialize Apollo
    apollo = Apollo(BOOK_PATH)
    apollo_move_count = 0

    # Initialize the chess engine
    with chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe") as engine:
        board = chess.Board()

        # Play game
        while not board.is_game_over():
            print("\nCurrent position:")
            print(board)

            if board.turn == chess.WHITE:
                # Apollo's turn
                fen = board.fen()
                move = apollo.get_move(fen)
                if move:
                    print(f"Apollo's move: {move}")
                    board.push_uci(move)
                    apollo_move_count += 1
                else:
                    print("Apollo has no move in the book. Ending test.")
                    break
            else:
                # Engine's turn
                result = engine.play(board, chess.engine.Limit(time=1))
                board.push(result.move)
                print(f"Engine's move: {result.move}")

        print("\nGame over!")
        print(f"Result: {board.result()}")
        print(f"Apollo made {apollo_move_count} moves.")


if __name__ == "__main__":
    play_against_engine()

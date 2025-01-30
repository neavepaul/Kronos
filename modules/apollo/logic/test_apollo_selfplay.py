import chess
from apollo import Apollo

# Path to the Polyglot book
BOOK_PATH = "data/merged_book.bin"

# Mapping of ECO codes to opening names (simplified example)
OPENING_NAMES = {
    "C50": "Italian Game",
    "B20": "Sicilian Defense",
    "E4": "King's Pawn Opening",
    # Add more mappings as needed
}

def identify_opening(board):
    """
    Identify the opening name from the current position.
    :param board: A chess.Board object representing the current position.
    :return: Opening name or 'Unknown Opening'.
    """
    eco_code = None
    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            entry = reader.find(board)
            eco_code = entry.weight  # Using `entry.weight` as a placeholder for ECO code
    except IndexError:
        pass

    return OPENING_NAMES.get(eco_code, "Unknown Opening") if eco_code else "Unknown Opening"


def self_play():
    """
    Simulate a self-play game where Apollo uses the opening book for both sides.
    """
    # Initialize Apollo
    apollo = Apollo(BOOK_PATH)
    board = chess.Board()

    move_counter = 0

    while not board.is_game_over():
        print("\nCurrent position:")
        print(board)

        # Apollo's turn
        fen = board.fen()
        move = apollo.get_move(fen)

        if move:
            opening_name = identify_opening(board)
            print(f"Move {move_counter + 1}: Apollo plays {move}")
            print(f"Current Opening: {opening_name}")
            board.push_uci(move)
        else:
            print("Apollo has no move in the book. Ending test.")
            break

        move_counter += 1

        # End after 20 moves or when out of book
        if move_counter >= 50:
            print("Test stopped after 20 moves to limit self-play.")
            break

    print("\nGame over!")
    print(f"Result: {board.result()}")
    print(f"Apollo made {move_counter} moves.")


if __name__ == "__main__":
    self_play()

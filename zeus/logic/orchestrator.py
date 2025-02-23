import chess
import chess.engine
import time
import numpy as np
import sys
from pathlib import Path

# 🛠️ Add Kronos root directory to sys.path
ROOT_PATH = Path(__file__).resolve().parents[2]  # Moves up to "Kronos/"
sys.path.append(str(ROOT_PATH))

# Import Modules
from modules.athena.src.athena import Athena
from modules.apollo.logic.apollo import Apollo
from modules.hades.logic.hades import Hades2

# 🔄 GUI Toggle: Set to False for headless mode (saves RAM)
ENABLE_GUI = True

if ENABLE_GUI:
    import pygame

# Paths
STOCKFISH_PATH = ROOT_PATH / "modules/shared/stockfish/stockfish-windows-x86-64-avx2.exe"
APOLLO_BOOK_PATH = ROOT_PATH / "modules/apollo/logic/data/merged_book.bin"
MODEL_PATH = ROOT_PATH / "modules/athena/src/models/athena_DQN_20250218_205619_200epochs.keras"
MOVE_VOCAB_PATH = ROOT_PATH / "modules/athena/src/move_vocab.json"

# Initialize Modules
apollo = Apollo(APOLLO_BOOK_PATH)
athena = Athena(MODEL_PATH, MOVE_VOCAB_PATH)
hades = Hades2()

use_apollo = True  # Keep using Apollo until he has no book moves

# 🎨 GUI Settings (Only if enabled)
if ENABLE_GUI:
    BOARD_SIZE = 600
    SQUARE_SIZE = BOARD_SIZE // 8
    INFO_HEIGHT = 100  # Space for Stockfish eval and move history
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE + INFO_HEIGHT))
    pygame.display.set_caption("Zeus: Apollo + Athena + Hades")

    # Colors
    WHITE = (238, 238, 210)
    BLACK = (118, 150, 86)
    BG_COLOR = (30, 30, 30)
    TEXT_COLOR = (255, 255, 255)
    BUTTON_COLOR = (70, 130, 180)

    # Load Font
    font = pygame.font.Font(None, 32)

    # Load Chess Piece Images
    pieces_images = {}
    piece_types = ["wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]
    for piece in piece_types:
        image = pygame.image.load(str(ROOT_PATH / f"modules/shared/assets/{piece}.png"))
        pieces_images[piece] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(board, eval_score=None, move_history=[]):
    """Draws an 8x8 chessboard with pieces and Stockfish evaluation if GUI is enabled."""
    if not ENABLE_GUI:
        return

    screen.fill(BG_COLOR)

    # Draw Board
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    # Draw Pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            piece_str = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().lower()}"
            screen.blit(pieces_images[piece_str], (col * SQUARE_SIZE, row * SQUARE_SIZE))

    # Draw Stockfish Evaluation
    eval_text = f"Stockfish Eval: {eval_score / 100:.2f}" if eval_score is not None else "Evaluating..."
    eval_surface = font.render(eval_text, True, TEXT_COLOR)
    screen.blit(eval_surface, (20, BOARD_SIZE + 10))

    # Draw Last 5 Moves
    history_text = "Moves: " + " ".join(move_history[-5:])
    history_surface = font.render(history_text, True, TEXT_COLOR)
    screen.blit(history_surface, (20, BOARD_SIZE + 40))

    pygame.display.flip()

def get_stockfish_eval(stockfish_engine, board):
    """Gets Stockfish evaluation for the board position."""
    eval_result = stockfish_engine.analyse(board, chess.engine.Limit(depth=10))
    return eval_result["score"].relative.score(mate_score=10000)

def count_pieces(board):
    """Counts the number of pieces for each color."""
    white_pieces = {"♙ Pawns": 0, "♘ Knights": 0, "♗ Bishops": 0, "♖ Rooks": 0, "♕ Queens": 0, "♔ King": 0}
    black_pieces = {"♟ Pawns": 0, "♞ Knights": 0, "♝ Bishops": 0, "♜ Rooks": 0, "♛ Queens": 0, "♚ King": 0}

    piece_symbols = {
        chess.PAWN: ("♙ Pawns", "♟ Pawns"),
        chess.KNIGHT: ("♘ Knights", "♞ Knights"),
        chess.BISHOP: ("♗ Bishops", "♝ Bishops"),
        chess.ROOK: ("♖ Rooks", "♜ Rooks"),
        chess.QUEEN: ("♕ Queens", "♛ Queens"),
        chess.KING: ("♔ King", "♚ King"),
    }

    for square, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            white_pieces[piece_symbols[piece.piece_type][0]] += 1
        else:
            black_pieces[piece_symbols[piece.piece_type][1]] += 1

    return white_pieces, black_pieces

def play_game():
    """Runs the game loop where Zeus decides whether to use Apollo, Athena, or Hades."""
    global use_apollo  # Track if Apollo should still be used

    board = chess.Board()
    move_history = []
    athena_side = chess.WHITE if input("Choose Athena's side (white/black): ").strip().lower() == "white" else chess.BLACK

    # Start Stockfish
    with chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH)) as stockfish_engine:
        while not board.is_game_over():
            time.sleep(1)
            move = None
            move_source = "Stockfish"  # Default source

            # 🎭 **Apollo Opening Book (Use Until Empty)**
            if use_apollo and board.turn == athena_side:
                move = apollo.get_opening_move(board)
                if move:
                    move_source = "📖 Apollo"
                else:
                    use_apollo = False  # Stop using Apollo when no book move is found

            # ⚔️ **Hades Endgame (7 or fewer pieces)**
            if move is None and len(board.piece_map()) <= 7 and board.turn == athena_side:
                move = hades.get_best_endgame_move(board)
                if move:
                    move_source = "🔥 Hades"

            # 🧠 **Athena Midgame (Neural Network)**
            if move is None and board.turn == athena_side:
                print("♟️ Athena to move")

                turn_indicator = 1 if board.turn == chess.WHITE else 0  # Set turn indicator
                legal_moves_mask = np.zeros((64, 64))  # Initialize legal move mask
                for legal_move in board.legal_moves:
                    legal_moves_mask[legal_move.from_square, legal_move.to_square] = 1  # Mark legal moves

                # Get move prediction from Athena
                move = athena.predict_move(board.fen(), move_history, legal_moves_mask, turn_indicator)
                move_source = "🧠 Athena"

            # ♜ **Stockfish Plays If No Other Option**
            if move is None:
                move = stockfish_engine.play(board, chess.engine.Limit(time=0.1)).move
                move_source = "♜ Stockfish"

            # Push move and update history
            board.push(move)
            move_history.append(move.uci())

            # Get Stockfish evaluation
            eval_score = get_stockfish_eval(stockfish_engine, board)

            # Determine move color
            move_color = "⚪ White" if board.turn == chess.BLACK else "⚫ Black"

            # Get piece counts
            white_pieces, black_pieces = count_pieces(board)

            # 🖥️ Print move details in Terminal
            print(f"{move_source} {move_color} → {move.uci()} | 📊 Eval: {eval_score / 100:.2f}")

            # Print piece counts
            print("\n♔ White Pieces:")
            for piece, count in white_pieces.items():
                print(f"   {piece}: {count}")

            print("\n♚ Black Pieces:")
            for piece, count in black_pieces.items():
                print(f"   {piece}: {count}")
            print("-" * 50)

            # 🎨 Update GUI if enabled
            if ENABLE_GUI:
                draw_board(board, eval_score, move_history)

        print(f"🏁 Game Over! Result: {board.result()}")

        if ENABLE_GUI:
            pygame.time.wait(5000)

if __name__ == "__main__":
    play_game()

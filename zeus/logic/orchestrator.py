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
from modules.hades.logic.hades import Hades

# 🔄 GUI Toggle: Set to False for headless mode
ENABLE_GUI = False

if ENABLE_GUI:
    import pygame

# Paths
STOCKFISH_PATH = ROOT_PATH / "modules/shared/stockfish/stockfish-windows-x86-64-avx2.exe"
APOLLO_BOOK_PATH = ROOT_PATH / "modules/apollo/logic/data/merged_book.bin"
MODEL_PATH = ROOT_PATH / "modules/athena/src/models/athena_gm_trained_20250211_090344_10epochs.keras"
MOVE_VOCAB_PATH = ROOT_PATH / "modules/athena/src/move_vocab.json"
SYZYGY_PATH = ROOT_PATH / "modules/hades/logic/syzygy/"

# Initialize Modules
apollo = Apollo(APOLLO_BOOK_PATH)
athena = Athena(MODEL_PATH, MOVE_VOCAB_PATH)
hades = Hades(SYZYGY_PATH)

if ENABLE_GUI:
    # GUI Settings
    BOARD_SIZE = 600
    SQUARE_SIZE = BOARD_SIZE // 8
    INFO_HEIGHT = 100  # Space for Stockfish eval and history
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE + INFO_HEIGHT))
    pygame.display.set_caption("Zeus: Apollo + Athena + Hades")

    # Colors
    WHITE = (238, 238, 210)
    BLACK = (118, 150, 86)
    BUTTON_COLOR = (70, 130, 180)
    TEXT_COLOR = (255, 255, 255)
    BG_COLOR = (30, 30, 30)

    # Font
    font = pygame.font.Font(None, 32)

def draw_board(board, eval_score=None, move_history=[]):
    """Draws an 8x8 chessboard with pieces and Stockfish evaluation if GUI is enabled."""
    if not ENABLE_GUI:
        return

    screen.fill(BG_COLOR)

    # Draw board
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    # Draw pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            piece_str = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().lower()}"
            image_path = ROOT_PATH / f"modules/shared/assets/{piece_str}.png"
            piece_image = pygame.image.load(str(image_path))
            piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))
            screen.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    # Draw evaluation score
    eval_text = f"Stockfish Eval: {eval_score / 100:.2f}" if eval_score is not None else "Evaluating..."
    eval_surface = font.render(eval_text, True, TEXT_COLOR)
    screen.blit(eval_surface, (20, BOARD_SIZE + 10))

    # Draw last 5 moves
    history_text = "Moves: " + " ".join(move_history[-5:])
    history_surface = font.render(history_text, True, TEXT_COLOR)
    screen.blit(history_surface, (20, BOARD_SIZE + 40))

    pygame.display.flip()

def draw_selection_screen():
    """Draws the screen for selecting Athena's side. Skips if GUI is disabled."""
    if not ENABLE_GUI:
        athena_side = input("Choose Athena's side (white/black): ").strip().lower()
        return chess.WHITE if athena_side == "white" else chess.BLACK

    screen.fill(BG_COLOR)

    text = font.render("Select Athena's Side", True, TEXT_COLOR)
    screen.blit(text, (BOARD_SIZE // 2 - text.get_width() // 2, 150))

    button_rects = {
        "white": pygame.Rect(50, 250, 200, 50),
        "black": pygame.Rect(350, 250, 200, 50),
    }

    for key, rect in button_rects.items():
        pygame.draw.rect(screen, BUTTON_COLOR, rect)
        text = font.render(key.capitalize(), True, TEXT_COLOR)
        screen.blit(text, (rect.x + 50, rect.y + 10))

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rects["white"].collidepoint(event.pos):
                    return chess.WHITE
                if button_rects["black"].collidepoint(event.pos):
                    return chess.BLACK

def get_stockfish_eval(stockfish_engine, board):
    """Gets Stockfish evaluation for the board position."""
    eval_result = stockfish_engine.analyse(board, chess.engine.Limit(depth=10))
    return eval_result["score"].relative.score(mate_score=10000)

def play_game():
    """Runs the game loop where Zeus decides whether to use Apollo, Athena, or Hades."""
    board = chess.Board()
    move_history = []
    athena_side = draw_selection_screen()  # Choose side for Athena

    # Start Stockfish
    with chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH)) as stockfish_engine:
        while not board.is_game_over():
            draw_board(board, None, move_history)
            time.sleep(1)
            move = None

            # 1️⃣ Apollo: Use opening book for the first 10 moves
            if len(move_history) < 10:
                move = apollo.get_opening_move(board)

            # 2️⃣ Hades: Check for endgame (5 or fewer pieces)
            if move is None and len(board.piece_map()) <= 5:
                move = hades.get_best_endgame_move(board)

            # 3️⃣ Athena: If no Apollo or Hades move, predict using NN
            if move is None and board.turn == athena_side:
                eval_score = get_stockfish_eval(stockfish_engine, board)
                move = athena.predict_move(board, move_history, eval_score)

            # 4️⃣ Stockfish plays if it's not Athena's turn
            if move is None:
                move = stockfish_engine.play(board, chess.engine.Limit(time=0.1)).move

            # Push move and update UI
            board.push(move)
            move_history.append(move.uci())

            # Get Stockfish evaluation after Athena’s move
            eval_score = get_stockfish_eval(stockfish_engine, board)

            if ENABLE_GUI:
                draw_board(board, eval_score, move_history)
            else:
                print(f"Move: {move.uci()} | Stockfish Eval: {eval_score / 100:.2f}")

        print(f"🏁 Game Over! Result: {board.result()}")
        hades.close()  # Close tablebase when done

        if ENABLE_GUI:
            pygame.time.wait(5000)

if __name__ == "__main__":
    play_game()

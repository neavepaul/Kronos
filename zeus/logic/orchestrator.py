import chess
import chess.engine
import chess.polyglot
import pygame
import time
import json
import numpy as np
import random
import tensorflow as tf
import sys
from pathlib import Path

# 🛠️ Add Kronos root directory to sys.path
ROOT_PATH = Path(__file__).resolve().parents[2]  # Moves up to "Kronos/"
sys.path.append(str(ROOT_PATH))

# ✅ Correct Imports from Modules
from modules.athena.src.model import get_model, TransformerBlock
from modules.athena.src.utils import fen_to_tensor, move_to_index

# Paths
STOCKFISH_PATH = ROOT_PATH / "modules/shared/stockfish/stockfish-windows-x86-64-avx2.exe"
APOLLO_BOOK_PATH = ROOT_PATH / "modules/apollo/logic/data/merged_book.bin"
MODEL_PATH = ROOT_PATH / "modules/athena/src/models/athena_gm_trained_20250211_090344_10epochs.keras"
MOVE_VOCAB_PATH = ROOT_PATH / "modules/athena/src/move_vocab.json"

# Load Athena model
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"TransformerBlock": TransformerBlock}  # ✅ Register custom layer
)

with open(MOVE_VOCAB_PATH, "r") as f:
    move_vocab = json.load(f)

# Load Apollo’s Opening Book
apollo_book = chess.polyglot.open_reader(str(APOLLO_BOOK_PATH))

# GUI Settings
BOARD_SIZE = 600
SQUARE_SIZE = BOARD_SIZE // 8
INFO_HEIGHT = 100  # Space for Stockfish eval and history
pygame.init()
screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE + INFO_HEIGHT))
pygame.display.set_caption("Zeus: Apollo + Athena")

# Colors
WHITE = (238, 238, 210)
BLACK = (118, 150, 86)
BUTTON_COLOR = (70, 130, 180)
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (30, 30, 30)

# Font
font = pygame.font.Font(None, 32)

# Load chess piece images
pieces_images = {}
piece_types = ["wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]
for piece in piece_types:
    image_path = ROOT_PATH / f"modules/shared/assets/{piece}.png"
    image = pygame.image.load(str(image_path))
    pieces_images[piece] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(board, eval_score=None, move_history=[]):
    """Draws an 8x8 chessboard with pieces and Stockfish evaluation."""
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
            screen.blit(pieces_images[piece_str], (col * SQUARE_SIZE, row * SQUARE_SIZE))

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
    """Draws the screen for selecting Athena's side."""
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

def apollo_predict_move(board):
    """Randomly picks a move from Apollo's opening book instead of always selecting the first move."""
    try:
        book_moves = list(apollo_book.find_all(board))
        if book_moves:
            chosen_move = random.choice(book_moves).move
            print(f"📖 Apollo Opening Move: {chosen_move}")
            return chosen_move
    except StopIteration:
        pass

    print("❌ No book move found in Apollo. Handing over to Athena.")
    return None

def athena_predict_move(fen, move_history, legal_moves, turn_indicator, eval_score):
    """Uses Athena's NN to predict the best move."""
    board = chess.Board(fen)

    # Convert inputs
    fen_tensor = np.expand_dims(fen_to_tensor(fen), axis=0)
    move_history_encoded = np.array([move_to_index(move_history, move_vocab)]).reshape(1, -1) if move_history else np.zeros((1, 50))
    legal_moves_array = np.array(legal_moves).reshape(1, 64, 64)
    turn_indicator_array = np.array([[turn_indicator]])
    eval_score_array = np.array([[eval_score]])

    # Model Prediction
    move_output, _ = model.predict([fen_tensor, move_history_encoded, legal_moves_array, turn_indicator_array, eval_score_array])
    move_index = np.argmax(move_output[0])

    legal_moves_list = list(board.legal_moves)
    if move_index < len(legal_moves_list):
        best_move = legal_moves_list[move_index]
    else:
        best_move = legal_moves_list[0]

    print(f"♟️ Athena Move: {best_move}")
    return best_move

def play_game():
    """Runs the game loop where Zeus decides whether to use Apollo or Athena."""
    board = chess.Board()
    move_history = []
    athena_side = draw_selection_screen()

    # Start Stockfish
    with chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH)) as stockfish_engine:
        while not board.is_game_over():
            draw_board(board, None, move_history)
            time.sleep(1)

            # Use Apollo for first 10 moves
            if len(move_history) < 10:
                move = apollo_predict_move(board)
                if move:
                    board.push(move)
                    move_history.append(move.uci())
                    draw_board(board, None, move_history)
                    continue

            if board.turn == athena_side:
                eval_score = get_stockfish_eval(stockfish_engine, board)
                legal_moves_mask = np.zeros((64, 64))
                for legal_move in board.legal_moves:
                    legal_moves_mask[legal_move.from_square, legal_move.to_square] = 1
                turn_indicator = 1 if board.turn == chess.WHITE else 0
                move = athena_predict_move(board.fen(), move_history, legal_moves_mask, turn_indicator, eval_score)
            else:
                move = stockfish_engine.play(board, chess.engine.Limit(time=0.1)).move

            board.push(move)
            move_history.append(move.uci())

        print(f"🏁 Game Over! Result: {board.result()}")
        pygame.time.wait(5000)

if __name__ == "__main__":
    play_game()

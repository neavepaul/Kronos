import chess
import chess.engine
import pygame
import time
import json
import numpy as np
import tensorflow as tf
from model import get_model
from utils import fen_to_tensor, move_to_index, build_move_vocab
from model import TransformerBlock  # Import your TransformerBlock class

# Paths to Stockfish binaries
STOCKFISH_PATH_1 = "stockfish/stockfish-windows-x86-64-avx2.exe"
STOCKFISH_PATH_2 = "stockfish/stockfish-windows-x86-64-avx2.exe"

# Load Athena model
MODEL_PATH = "models/athena_gm_trained_20250211_090344_10epochs.keras"
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"TransformerBlock": TransformerBlock}  # Register custom layer
)

# Load move vocabulary
with open("move_vocab.json", "r") as f:
    move_vocab = json.load(f)

# GUI Settings
BOARD_SIZE = 600
SQUARE_SIZE = BOARD_SIZE // 8
pygame.init()
screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
pygame.display.set_caption("Athena vs. Stockfish")

# Colors
WHITE = (238, 238, 210)
BLACK = (118, 150, 86)
BUTTON_COLOR = (70, 130, 180)
TEXT_COLOR = (255, 255, 255)

# Load and scale chess pieces
pieces_images = {}
piece_types = ["wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]
for piece in piece_types:
    image = pygame.image.load(f"assets/{piece}.png")
    pieces_images[piece] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

# Button for choosing who plays first
font = pygame.font.Font(None, 40)
button_rects = {
    "stockfish": pygame.Rect(50, 250, 200, 50),
    "athena": pygame.Rect(350, 250, 200, 50),
}

def draw_board(board, eval_score=None):
    """Draws an 8x8 chessboard with pieces and Stockfish evaluation."""
    screen.fill((30, 30, 30))  # Dark background

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

    # Draw Stockfish evaluation
    if eval_score is not None:
        eval_text = f"Stockfish Eval: {eval_score / 100:.2f}"  # Convert centipawns to pawns
        eval_surface = font.render(eval_text, True, TEXT_COLOR)
        screen.blit(eval_surface, (BOARD_SIZE // 2 - eval_surface.get_width() // 2, BOARD_SIZE + 20))

    pygame.display.flip()

def draw_selection_screen():
    """Draws the initial screen to choose who plays first."""
    screen.fill((30, 30, 30))
    
    text = font.render("Who plays first?", True, TEXT_COLOR)
    screen.blit(text, (BOARD_SIZE // 2 - text.get_width() // 2, 150))

    for key, rect in button_rects.items():
        pygame.draw.rect(screen, BUTTON_COLOR, rect)
        text = font.render(key.capitalize(), True, TEXT_COLOR)
        screen.blit(text, (rect.x + 50, rect.y + 10))

    pygame.display.flip()

def get_first_player():
    """Waits for user input to select first player."""
    while True:
        draw_selection_screen()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rects["stockfish"].collidepoint(event.pos):
                    return "stockfish"
                if button_rects["athena"].collidepoint(event.pos):
                    return "athena"

def get_stockfish_eval(stockfish_engine, board):
    """Get the Stockfish evaluation for a given board position."""
    eval_result = stockfish_engine.analyse(board, chess.engine.Limit(depth=10))
    return eval_result["score"].relative.score(mate_score=10000)  # Convert mate score properly

def athena_predict_move(fen, move_history, legal_moves, turn_indicator, eval_score):
    """Uses Athena to predict the best move."""
    fen_tensor = np.expand_dims(fen_to_tensor(fen), axis=0)
    move_history_encoded = np.array([move_to_index(move_history, move_vocab)]).reshape(1, -1) if move_history else np.zeros((1, 50))
    legal_moves_array = np.array(legal_moves).reshape(1, 64, 64)
    turn_indicator_array = np.array([[turn_indicator]])
    eval_score_array = np.array([[eval_score]])

    move_output, _ = model.predict([fen_tensor, move_history_encoded, legal_moves_array, turn_indicator_array, eval_score_array])
    print(f"📊 Athena Move Scores: {move_output[0]}")  # Print raw move scores

    
    move_index = np.argmax(move_output[0])
    legal_moves_list = list(chess.Board(fen).legal_moves)

    best_move = legal_moves_list[move_index] if move_index < len(legal_moves_list) else legal_moves_list[0]

    # Compare Athena's move to Stockfish
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH_2) as stockfish:
        sf_move = stockfish.play(chess.Board(fen), chess.engine.Limit(time=0.1)).move
        print(f"♟️ Athena Move: {best_move} | ⚡ Stockfish Move: {sf_move}")

    return best_move


def play_game():
    """Runs the game loop where Stockfish and Athena play, and another Stockfish evaluates."""
    board = chess.Board()
    move_history = []

    # Player selection
    first_player = get_first_player()
    print(f"{first_player.capitalize()} plays first!")

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH_1) as stockfish_1, \
         chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH_2) as stockfish_2:
        
        while not board.is_game_over():
            eval_score = None  # Initialize eval_score for display
            draw_board(board, eval_score)
            time.sleep(1)

            if (board.turn == chess.WHITE and first_player == "stockfish") or (board.turn == chess.BLACK and first_player == "athena"):
                # Stockfish Move
                result = stockfish_1.play(board, chess.engine.Limit(time=0.1))
                move = result.move
            else:
                # Get Stockfish eval before Athena moves
                eval_score = get_stockfish_eval(stockfish_2, board)

                # Athena Move
                legal_moves_mask = np.zeros((64, 64))
                for legal_move in board.legal_moves:
                    legal_moves_mask[legal_move.from_square, legal_move.to_square] = 1

                turn_indicator = 1 if board.turn == chess.WHITE else 0
                move = athena_predict_move(board.fen(), move_history, legal_moves_mask, turn_indicator, eval_score)

            # Push move and record in proper format
            board.push(move)
            move_encoded = move_to_index(move.uci(), move_vocab) 
            move_history.append(move_encoded)

            # Evaluate Athena's move using Stockfish 2
            eval_score = get_stockfish_eval(stockfish_2, board)
            print(f"Stockfish evaluation after Athena's move: {eval_score}")

        print(f"Game Over! Result: {board.result()}")
        draw_board(board, eval_score)
        pygame.time.wait(5000)

if __name__ == "__main__":
    play_game()

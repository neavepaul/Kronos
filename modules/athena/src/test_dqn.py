import chess
import chess.engine
import pygame
import time
import json
import numpy as np
import tensorflow as tf
from model import TransformerBlock
from utils import fen_to_tensor, move_to_index, build_move_vocab
from dqn import DQN  # Load DQN model

# Paths
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
MODEL_PATH = "models/athena_DQN_20250218_205619_200epochs.keras"

# Load Athena Model
dqn_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"TransformerBlock": TransformerBlock}  # Register custom layers if needed
)

# Load move vocabulary
with open("move_vocab.json", "r") as f:
    move_vocab = json.load(f)

# GUI Settings
BOARD_SIZE = 600
SQUARE_SIZE = BOARD_SIZE // 8
pygame.init()
screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
pygame.display.set_caption("Athena DQN vs. Stockfish")

# Colors
WHITE = (238, 238, 210)
BLACK = (118, 150, 86)
TEXT_COLOR = (255, 255, 255)

# Load and scale chess pieces
pieces_images = {}
piece_types = ["wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]
for piece in piece_types:
    image = pygame.image.load(f"assets/{piece}.png")
    pieces_images[piece] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

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
        font = pygame.font.Font(None, 40)
        eval_text = f"Stockfish Eval: {eval_score / 100:.2f}"  # Convert centipawns to pawns
        eval_surface = font.render(eval_text, True, TEXT_COLOR)
        screen.blit(eval_surface, (BOARD_SIZE // 2 - eval_surface.get_width() // 2, BOARD_SIZE + 20))

    pygame.display.flip()

def get_stockfish_eval(stockfish_engine, board):
    """Get the Stockfish evaluation for a given board position."""
    eval_result = stockfish_engine.analyse(board, chess.engine.Limit(depth=10))
    return eval_result["score"].relative.score(mate_score=10000)  # Convert mate score properly

def athena_predict_move(fen, move_history, legal_moves, turn_indicator):
    """Uses the trained Athena DQN model to predict the best move."""
    fen_tensor = np.expand_dims(fen_to_tensor(fen), axis=0)
    move_history_encoded = np.array([move_to_index(move_history, move_vocab)]).reshape(1, -1) if move_history else np.zeros((1, 50))
    legal_moves_array = np.array(legal_moves).reshape(1, 64, 64)
    turn_indicator_array = np.array([[turn_indicator]])

    q_values = dqn_model([fen_tensor, move_history_encoded, legal_moves_array, turn_indicator_array])
    
    move_index = np.argmax(q_values.numpy()[0])
    legal_moves_list = list(chess.Board(fen).legal_moves)

    best_move = legal_moves_list[move_index] if move_index < len(legal_moves_list) else legal_moves_list[0]

    # Compare Athena's move to Stockfish
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
        sf_move = stockfish.play(chess.Board(fen), chess.engine.Limit(time=0.1)).move
        print(f"♟️ Athena Move: {best_move} | ⚡ Stockfish Move: {sf_move}")

    return best_move

def play_game():
    """Runs a **single game** between Athena and Stockfish."""
    board = chess.Board()
    move_history = []

    # User selects Stockfish difficulty
    skill_level = int(input("Enter Stockfish skill level (0-20): ").strip())

    print(f"\n🌟 Starting Game | Stockfish Skill Level: {skill_level}")
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
        stockfish.configure({"Skill Level": skill_level})

        while not board.is_game_over():
            eval_score = get_stockfish_eval(stockfish, board)
            draw_board(board, eval_score)
            time.sleep(1)

            # Athena's move
            if board.turn == chess.WHITE:
                print("♟️ White to move (Athena)")
                turn_indicator = 1
                legal_moves_mask = np.zeros((64, 64))
                for legal_move in board.legal_moves:
                    legal_moves_mask[legal_move.from_square, legal_move.to_square] = 1

                athena_move = athena_predict_move(board.fen(), move_history, legal_moves_mask, turn_indicator)
                board.push(athena_move)
            else:
                # Stockfish's move
                print("⚡ Black to move (Stockfish)")
                result = stockfish.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)

            # Store move history
            move_history.append(board.peek().uci())

            # Show updated evaluation
            eval_score = get_stockfish_eval(stockfish, board)
            print(f"📊 Stockfish Eval After Move: {eval_score}")
            draw_board(board, eval_score)

    print(f"🏁 Game Over! Result: {board.result()}")
    draw_board(board)
    pygame.time.wait(5000)

if __name__ == "__main__":
    play_game()

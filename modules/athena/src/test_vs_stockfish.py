import chess
import chess.engine
import pygame
import time
import numpy as np
from model import get_model
from utils import fen_to_tensor, move_to_index, build_move_vocab

# Paths to Stockfish binaries
STOCKFISH_PATH_1 = "stockfish/stockfish-windows-x86-64-avx2.exe"
STOCKFISH_PATH_2 = "stockfish/stockfish-windows-x86-64-avx2.exe"

# Load Athena model
model = get_model()
model.load_weights("models/athena_trained.h5")

# Load move vocabulary
move_vocab = build_move_vocab("training_data/")

# GUI Settings
BOARD_SIZE = 600
SQUARE_SIZE = BOARD_SIZE // 8
pygame.init()
screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
pygame.display.set_caption("Athena vs. Stockfish")

# Colors
WHITE = (238, 238, 210)
BLACK = (118, 150, 86)

# Load and scale chess pieces
pieces_images = {}
piece_types = ["wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]
for piece in piece_types:
    image = pygame.image.load(f"assets/{piece}.png")
    pieces_images[piece] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(board):
    """
    Draws an 8x8 chessboard with scaled pieces.
    """
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            piece_str = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().lower()}"
            screen.blit(pieces_images[piece_str], (col * SQUARE_SIZE, row * SQUARE_SIZE))

    pygame.display.flip()

def athena_predict_move(fen, move_history, legal_moves):
    """
    Use Athena to predict the best move.
    """
    fen_tensor = np.expand_dims(fen_to_tensor(fen), axis=0)

    # 🔥 Fix: Ensure move history is valid
    move_history_encoded = np.array([move_to_index(move_history, move_vocab)]).reshape(1, -1) if move_history else np.zeros((1, 50))

    legal_moves_array = np.array(legal_moves).reshape(1, 64, 64)

    eval_score, move_from, move_to = model.predict([fen_tensor, move_history_encoded, legal_moves_array])
    best_from = np.argmax(move_from)
    best_to = np.argmax(move_to)

    return chess.Move(best_from, best_to)

def play_game():
    """
    Runs the game loop where Stockfish and Athena play, and another Stockfish evaluates.
    """
    board = chess.Board()
    move_history = []
    
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH_1) as stockfish_1, \
         chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH_2) as stockfish_2:
        
        while not board.is_game_over():
            draw_board(board)
            time.sleep(1)  # Slow down for visualization
            
            if board.turn == chess.WHITE:
                # Stockfish 1 (White) makes a move
                result = stockfish_1.play(board, chess.engine.Limit(time=0.1))
                move = result.move
            else:
                # Athena (Black) predicts move
                legal_moves_mask = np.zeros((64, 64))
                for legal_move in board.legal_moves:
                    legal_moves_mask[legal_move.from_square, legal_move.to_square] = 1

                move = athena_predict_move(board.fen(), move_history, legal_moves_mask)

            board.push(move)
            move_history.append(move.uci())

            # Evaluate Athena's move using Stockfish 2
            if board.turn == chess.BLACK:
                eval_result = stockfish_2.analyse(board, chess.engine.Limit(depth=10))
                eval_score = eval_result["score"].relative.score(mate_score=10000)
                print(f"Stockfish evaluation after Athena's move: {eval_score}")

        print(f"Game Over! Result: {board.result()}")
        draw_board(board)
        pygame.time.wait(5000)

if __name__ == "__main__":
    play_game()

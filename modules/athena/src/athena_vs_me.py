import pygame
import chess
import sys
import os
import numpy as np
from tensorflow.keras.models import load_model

from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_PATH))
# Import your model
from modules.athena.src.aegis_net import AegisNet
from modules.athena.src.utils import fen_to_tensor, encode_move_sequence, get_attack_defense_maps

# --- Config ---
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
FPS = 60

# Paths
ASSETS_PATH = "assets/"
MODEL_PATH = str(ROOT_PATH / "modules/athena/src/models/athena_hybrid_final_20250510_181337_prometheus-20-50_0.keras")

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initialize model
# model = AegisNet()
# model.alpha_model.load_weights(MODEL_PATH)


model = load_model(MODEL_PATH)

# Load images
def load_piece_images():
    pieces = {}
    for piece in ['bp', 'bn', 'bb', 'br', 'bq', 'bk', 'wp', 'wn', 'wb', 'wr', 'wq', 'wk']:
        img = pygame.image.load(os.path.join(ASSETS_PATH, piece + ".png"))
        img = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        pieces[piece] = img
    return pieces

# Draw board
def draw_board(screen, board, images, selected_square=None, legal_moves=None):
    for rank in range(8):
        for file in range(8):
            color = WHITE if (rank + file) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, pygame.Rect(file*SQUARE_SIZE, rank*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            piece = board.piece_at(chess.square(file, 7 - rank))
            if piece:
                img_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().lower()
                screen.blit(images[img_key], (file*SQUARE_SIZE, rank*SQUARE_SIZE))

    # Highlight selected square
    if selected_square is not None:
        file = chess.square_file(selected_square)
        rank = 7 - chess.square_rank(selected_square)
        pygame.draw.rect(screen, GREEN, (file*SQUARE_SIZE, rank*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)

    # Highlight legal moves
    if legal_moves:
        for move in legal_moves:
            file = chess.square_file(move.to_square)
            rank = 7 - chess.square_rank(move.to_square)
            center = (file*SQUARE_SIZE + SQUARE_SIZE//2, rank*SQUARE_SIZE + SQUARE_SIZE//2)
            pygame.draw.circle(screen, BLUE, center, 10)

# Get board click
def get_square_under_mouse(pos):
    x, y = pos
    file = x // SQUARE_SIZE
    rank = 7 - (y // SQUARE_SIZE)
    return chess.square(file, rank)

# Model move
def model_move(board):
    fen_tensor = np.expand_dims(fen_to_tensor(board), axis=0)
    move_history = encode_move_sequence([m.uci() for m in board.move_stack])
    move_history = np.expand_dims(move_history, axis=0)
    attack_map, defense_map = get_attack_defense_maps(board)
    attack_map = np.expand_dims(attack_map, axis=0)
    defense_map = np.expand_dims(defense_map, axis=0)

    policy, _ = model.alpha_model.predict([fen_tensor, move_history, attack_map, defense_map], verbose=0)
    legal_moves = list(board.legal_moves)

    move_probs = {}
    for move in legal_moves:
        idx = move.from_square * 64 + move.to_square
        move_probs[move] = policy[0][idx]

    if move_probs:
        best_move = max(move_probs.items(), key=lambda x: x[1])[0]
        board.push(best_move)


def choose_promotion():
    """Popup for choosing promotion piece. Returns one of: chess.QUEEN, ROOK, BISHOP, KNIGHT"""
    popup_width, popup_height = 400, 100
    popup = pygame.display.set_mode((popup_width, popup_height))
    pygame.display.set_caption("Choose Promotion")

    font = pygame.font.SysFont(None, 48)
    text = font.render("Promote to:", True, (0, 0, 0))

    options = [
        ("Queen", chess.QUEEN),
        ("Rook", chess.ROOK),
        ("Bishop", chess.BISHOP),
        ("Knight", chess.KNIGHT)
    ]

    running = True
    while running:
        popup.fill((240, 240, 240))
        popup.blit(text, (20, 20))

        for idx, (name, _) in enumerate(options):
            pygame.draw.rect(popup, (200, 200, 200), (20 + idx*90, 60, 80, 30))
            label = font.render(name[0], True, (0, 0, 0))
            popup.blit(label, (35 + idx*90, 60))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                for idx, (_, piece_type) in enumerate(options):
                    if 20 + idx*90 <= x <= 20 + idx*90 + 80 and 60 <= y <= 90:
                        running = False
                        return piece_type

    return chess.QUEEN  # fallback if something goes wrong

def choose_color_screen():
    """Popup for choosing to play as White or Black. Returns True if playing White, False if Black."""
    popup_width, popup_height = 400, 300
    popup = pygame.display.set_mode((popup_width, popup_height))
    pygame.display.set_caption("Choose Side")

    font = pygame.font.SysFont(None, 40)
    title_font = pygame.font.SysFont(None, 50)

    running = True
    while running:
        popup.fill((240, 240, 240))

        # Draw title
        title_text = title_font.render("Choose Your Side", True, (0, 0, 0))
        popup.blit(title_text, (popup_width//2 - title_text.get_width()//2, 50))

        # Draw White button
        white_button = pygame.Rect(70, 150, 120, 50)
        pygame.draw.rect(popup, (200, 200, 200), white_button)
        white_text = font.render("White", True, (0, 0, 0))
        popup.blit(white_text, (white_button.centerx - white_text.get_width()//2, white_button.centery - white_text.get_height()//2))

        # Draw Black button
        black_button = pygame.Rect(210, 150, 120, 50)
        pygame.draw.rect(popup, (200, 200, 200), black_button)
        black_text = font.render("Black", True, (0, 0, 0))
        popup.blit(black_text, (black_button.centerx - black_text.get_width()//2, black_button.centery - black_text.get_height()//2))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if white_button.collidepoint(x, y):
                    return True
                elif black_button.collidepoint(x, y):
                    return False

def show_game_over_screen(board, play_as_white):
    """Popup for showing correct game result based on player color."""
    popup_width, popup_height = 400, 200
    popup = pygame.display.set_mode((popup_width, popup_height))
    pygame.display.set_caption("Game Over")

    font = pygame.font.SysFont(None, 48)

    player_color = chess.WHITE if play_as_white else chess.BLACK
    result = board.result()

    if (result == '1-0' and player_color == chess.WHITE) or (result == '0-1' and player_color == chess.BLACK):
        text = font.render("You Win!", True, (0, 128, 0))
    elif (result == '1-0' and player_color == chess.BLACK) or (result == '0-1' and player_color == chess.WHITE):
        text = font.render("Athena Wins!", True, (200, 0, 0))
    else:
        text = font.render("Draw...", True, (0, 0, 128))

    sub_font = pygame.font.SysFont(None, 32)
    subtext = sub_font.render("Click anywhere to quit", True, (100, 100, 100))

    running = True
    while running:
        popup.fill((245, 245, 245))
        popup.blit(text, (popup_width//2 - text.get_width()//2, 60))
        popup.blit(subtext, (popup_width//2 - subtext.get_width()//2, 120))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                running = False
                pygame.quit()
                sys.exit()


# --- Main ---
def main():
    pygame.init()

    play_as_white = choose_color_screen()

    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    pygame.display.set_caption("Play Against Athena")
    clock = pygame.time.Clock()
    images = load_piece_images()

    board = chess.Board()
    selected_square = None
    legal_moves = []

    # If player chose black, let Athena make the first move
    if not play_as_white:
        model_move(board)

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                square = get_square_under_mouse(pos)

                if selected_square is None:
                    if board.piece_at(square) and board.color_at(square) == (chess.WHITE if play_as_white else chess.BLACK):
                        selected_square = square
                        legal_moves = [move for move in board.legal_moves if move.from_square == selected_square]
                else:
                    move = chess.Move(selected_square, square)
                    piece = board.piece_at(selected_square)

                    if piece and piece.piece_type == chess.PAWN:
                        target_rank = chess.square_rank(square)
                        if (piece.color == chess.WHITE and target_rank == 7) or (piece.color == chess.BLACK and target_rank == 0):
                            promotion_piece = choose_promotion()
                            move = chess.Move(selected_square, square, promotion=promotion_piece)

                    if move in board.legal_moves:
                        board.push(move)
                        selected_square = None
                        legal_moves = []
                        pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))

                        if board.is_game_over():
                            show_game_over_screen(board, play_as_white)

                        if not board.is_game_over():
                            model_move(board)

                            if board.is_game_over():
                                show_game_over_screen(board, play_as_white)
                    else:
                        selected_square = None
                        legal_moves = []

        draw_board(screen, board, images, selected_square, legal_moves)
        pygame.display.flip()

if __name__ == "__main__":
    main()

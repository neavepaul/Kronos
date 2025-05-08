import chess
import chess.engine
import random
import sys
import os
import numpy as np
from pathlib import Path

# --- Config ---
ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_PATH))
from modules.athena.src.aegis_net import AegisNet
from modules.athena.src.utils import fen_to_tensor, encode_move_sequence, get_attack_defense_maps

CHOSEN_LEVEL = 0  # <<< Set your desired Stockfish skill level here (0-20)
PLAY_AS_WHITE = True  # Set to False to play as Black
STOCKFISH_PATH = str(Path(__file__).resolve().parents[3] / "modules/shared/stockfish/stockfish-windows-x86-64-avx2.exe")
MODEL_PATH = str(Path(__file__).resolve().parents[3] / "modules/athena/src/models/athena_hybrid_final_20250507_191400_elo_0.weights.h5")

# --- Load model ---
model = AegisNet()
model.alpha_model.load_weights(MODEL_PATH)

# --- Stockfish setup ---
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
engine.configure({'Skill Level': CHOSEN_LEVEL})

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

    return max(move_probs.items(), key=lambda x: x[1])[0] if move_probs else random.choice(legal_moves)

# --- Play one game ---
board = chess.Board()
player_color = chess.WHITE if PLAY_AS_WHITE else chess.BLACK

while not board.is_game_over():
    move = model_move(board) if board.turn == player_color else engine.play(board, chess.engine.Limit(time=0.05)).move
    board.push(move)

print("\nFinal Result:", board.result())
print(board)

# Print who played which side
your_color = "White" if PLAY_AS_WHITE else "Black"
print(f"\nYou played as: {your_color}")

# Determine winner
result = board.result()
if result == "1-0":
    print("White won the game.")
elif result == "0-1":
    print("Black won the game.")
else:
    print("The game was a draw.")

engine.quit()

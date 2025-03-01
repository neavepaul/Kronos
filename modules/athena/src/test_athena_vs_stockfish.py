import chess
import chess.engine
import numpy as np
import random
import sys
from pathlib import Path
from datetime import datetime

import tensorflow as tf

# Dynamically set the root path to the "Kronos" directory
ROOT_PATH = Path(__file__).resolve().parents[3]  # Moves up to "Kronos/"
sys.path.append(str(ROOT_PATH))

from modules.athena.src.utils import fen_to_tensor, move_to_index, encode_move, get_attack_defense_maps
from modules.athena.src.actor_critic import ActorCritic

# Paths
STOCKFISH_PATH = ROOT_PATH / "modules/shared/stockfish/stockfish-windows-x86-64-avx2.exe"
MODEL_PATH = ROOT_PATH / "modules/athena/src/models/athena_PPO_20250301_131113_200epochs.keras"

# Load the trained model
input_shape = (8, 8, 20)
action_size = 64 * 64
actor_critic_model = ActorCritic(input_shape, action_size)
actor_critic_model.load(MODEL_PATH)

def test_athena_vs_stockfish():
    """Test Athena against Stockfish at level 3."""
    board = chess.Board()
    move_history = []
    move_count = 0

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
        stockfish.configure({"Skill Level": 3})  # Set Stockfish skill level to 3

        while not board.is_game_over():
            move_count += 1
            fen_before = board.fen()
            attack_map, defense_map = get_attack_defense_maps(board)

            fen_tensor = np.expand_dims(fen_to_tensor(fen_before), axis=0)
            move_history_encoded = np.array(move_to_index(move_history, move_vocab), dtype=np.int32).reshape(1, 50)
            attack_map = np.expand_dims(attack_map, axis=(0, -1))
            defense_map = np.expand_dims(defense_map, axis=(0, -1))
            turn_indicator = np.array([[1.0 if board.turn == chess.WHITE else 0.0]], dtype=np.float32)

            state = (fen_tensor, move_history_encoded, attack_map, defense_map, turn_indicator)

            # Ensure legal_moves is a list of chess.Move objects
            legal_moves_list = list(board.legal_moves)

            if board.turn == chess.WHITE:
                action = actor_critic_model.sample_action(state, legal_moves_list)
            else:
                stockfish_move = stockfish.play(board, chess.engine.Limit(time=0.1)).move
                if stockfish_move in legal_moves_list:
                    action = stockfish_move
                else:
                    print("⚠️ Stockfish tried an illegal move. Selecting a random legal move.")
                    action = random.choice(legal_moves_list)  # Fallback safety

            # Final Check Before Execution
            if action not in legal_moves_list:
                print(f"⚠️ Attempted illegal move: {action}. Selecting a random legal move.")
                action = random.choice(legal_moves_list)

            # Execute Move
            board.push(action)  # No more illegal moves!
            move_history.append(action.uci())

            print(f"Move {move_count}: {action.uci()} ({'Athena' if board.turn == chess.BLACK else 'Stockfish'})")

    print(f"Game Over. Result: {board.result()}")
    if board.result() == "1-0":
        print("🎉 ATHENA WINS! 🎉")
    elif board.result() == "0-1":
        print("💔 STOCKFISH WINS! 💔")
    else:
        print("🤝 DRAW! 🤝")

if __name__ == "__main__":
    test_athena_vs_stockfish()
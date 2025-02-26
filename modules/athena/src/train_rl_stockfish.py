import tensorflow as tf
import numpy as np
import logging
import chess
import chess.engine
import random
import json
from datetime import datetime
from model import get_model
import os
import sys
from pathlib import Path

# Dynamically set the root path to the "Kronos" directory
ROOT_PATH = Path(__file__).resolve().parents[3]  # Moves up to "Kronos/"
sys.path.append(str(ROOT_PATH))

from modules.athena.src.reward_function import compute_reward
from modules.athena.src.actor_critic import ActorCritic, train_actor_critic
from modules.athena.src.replay_buffer import init_replay_buffer
from modules.athena.src.utils import fen_to_tensor, move_to_index, encode_move, get_attack_defense_maps, get_stockfish_eval_train


if os.path.exists(ROOT_PATH / "modules/athena/src/replay_buffer.h5"):
    os.remove("replay_buffer.h5")
    print("🚀 Corrupt replay buffer deleted! Restart training.")

# Load move vocabulary
MAX_VOCAB_SIZE = 500000
with open(ROOT_PATH / "modules/athena/src/move_vocab.json", "r") as f:
    move_vocab = json.load(f)

# RL Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 32
EPOCHS = 200

# Paths
STOCKFISH_PATH = ROOT_PATH / "modules/shared/stockfish/stockfish-windows-x86-64-avx2.exe"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = ROOT_PATH / f"modules/athena/src/models/athena_PPO_{timestamp}_{EPOCHS}epochs.keras"
# Setup logging
LOG_FILE = ROOT_PATH / "modules/athena/src/training_log.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize Model
input_shape = (8, 8, 20)
action_size = 64 * 64
actor_critic_model = ActorCritic(input_shape, action_size)

# Experience Replay Buffer
replay_buffer = init_replay_buffer()

def play_vs_stockfish(skill_level, athena_is_white, epoch):
    """Athena plays against Stockfish with increasing difficulty."""
    board = chess.Board()
    move_history = []
    move_count = 0 

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
        stockfish.configure({"Skill Level": skill_level})  

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

            if (board.turn == chess.WHITE and athena_is_white) or (board.turn == chess.BLACK and not athena_is_white):
                # Check if the position exists in Game Memory
                past_game = replay_buffer.game_memory.find_similar_position(board)
                if past_game:
                    print("♟️ Found similar position in memory. Recalling pattern.")
                    suggested_move = chess.Move.from_uci(past_game["moves"][0])
                else:
                    suggested_move = None

                action = actor_critic_model.sample_action(state, legal_moves_list)

                # If a past move exists in memory, adjust probability
                if suggested_move and suggested_move in legal_moves_list:
                    action = suggested_move
            else:
                stockfish_move = stockfish.play(board, chess.engine.Limit(time=0.1))
                action = stockfish_move.move

            board_before = board.copy()
            board.push(action)
            move_history.append(action.uci())
            fen_after = board.fen()
            fen_tensor_after = np.expand_dims(fen_to_tensor(fen_after), axis=0)
            
            eval_before = get_stockfish_eval_train(fen_before)
            eval_after = get_stockfish_eval_train(fen_after)
            eval_change = eval_after - eval_before

            game_result = None
            if board.is_game_over():
                outcome = board.outcome()
                if outcome.winner is None:
                    game_result = 0  # Draw
                elif outcome.winner == chess.WHITE:
                    game_result = 1  # White wins
                else:
                    game_result = -1  # Black wins

            reward = compute_reward(board_before, board, action, game_result)
            next_state = (fen_tensor_after, move_history_encoded, attack_map, defense_map, turn_indicator)

            encoded_move = encode_move(board_before, action)
            replay_buffer.add(state, encoded_move, reward, next_state, board.is_game_over(), eval_change, fen_after)
            print(f"Move {move_count}: {action.uci()} ({'Athena' if (board.turn != athena_is_white) else 'Stockfish'}) | Eval Change: {eval_change:.2f}")
            logging.info(f"Move {move_count}: {action.uci()} ({'Athena' if (board.turn != athena_is_white) else 'Stockfish'}) | Eval Change: {eval_change:.2f}")

    print(f"Game Over. Result: {board.result()}")
    logging.info(f"Game Over. Result: {board.result()}")

def train_athena():
    """Train Athena using PPO with graph-based encoding."""
    print("\n Training Athena Started...\n")
    logging.info("Training Athena Started...")
    
    for epoch in range(EPOCHS):
        progress = epoch / EPOCHS
        if progress < 0.25:    # 0% - 25%
            skill_level = random.randint(0, 3)  # Beginner
        elif progress < 0.50:  # 25% - 50%
            skill_level = random.randint(4, 7)  # Intermediate
        elif progress < 0.75:  # 50% - 75%
            skill_level = random.randint(8, 12)  # Advanced
        elif progress < 0.90:  # 75% - 90%
            skill_level = random.randint(13, 16)  # Strong
        else:                  # Last 10%
            skill_level = random.randint(17, 20)  # Elite

        athena_is_white = bool(random.getrandbits(1))  # Randomly assign color

        print(f"\n🟢 Epoch {epoch + 1}/{EPOCHS} | Stockfish Skill: {skill_level} | Athena {'White' if athena_is_white else 'Black'}")
        logging.info(f"Epoch {epoch + 1}/{EPOCHS} | Stockfish Skill: {skill_level} | Athena {'White' if athena_is_white else 'Black'}")

        play_vs_stockfish(skill_level, athena_is_white, epoch)
        loss = train_actor_critic(actor_critic_model, replay_buffer, BATCH_SIZE)

        if loss is not None:
            print(f"Training Loss: {loss:.6f}")
            logging.info(f"📉 Training Loss: {loss:.6f}")
        else:
            print("⚠️ Skipping training - Not enough data in replay buffer.")
            logging.warning("⚠️ Skipping training - Not enough data in replay buffer.")

    # Save model at the end
    actor_critic_model.save(MODEL_SAVE_PATH)
    print(f"\nTraining Complete! Model saved at: {MODEL_SAVE_PATH}")
    logging.info(f"\n💾 Training Complete! Model saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_athena()

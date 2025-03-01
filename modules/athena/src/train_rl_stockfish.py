import json
import logging
import chess
import chess.engine
import random
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Dynamically set the root path to the "Kronos" directory
ROOT_PATH = Path(__file__).resolve().parents[3]  # Moves up to "Kronos/"
sys.path.append(str(ROOT_PATH))

from modules.athena.src.reward_function import compute_reward
from modules.athena.src.actor_critic import ActorCritic, train_actor_critic
from modules.apollo.logic.apollo import Apollo
from modules.hades.logic.hades import Hades2
from modules.athena.src.replay_buffer import init_replay_buffer
from modules.athena.src.utils import fen_to_tensor, encode_move, get_attack_defense_maps


if os.path.exists(ROOT_PATH / "modules/athena/src/replay_buffer.h5"):
    os.remove("replay_buffer.h5")
    print("🚀 Corrupt replay buffer deleted! Restart training.")

# RL Hyperparameters
BATCH_SIZE = 64
EPOCHS = 200

# Paths
APOLLO_BOOK_PATH = ROOT_PATH / "modules/apollo/logic/data/merged_book.bin"
STOCKFISH_PATH = ROOT_PATH / "modules/shared/stockfish/stockfish-windows-x86-64-avx2.exe"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = ROOT_PATH / f"modules/athena/src/models/athena_PPO_{timestamp}_{EPOCHS}epochs.keras"
# Setup logging
LOG_FILE = ROOT_PATH / "modules/athena/src/training_log.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

apollo = Apollo(APOLLO_BOOK_PATH)
hades = Hades2()  # Initialize Hades

# Initialize Model
input_shape = (8, 8, 20)
action_size = 64 * 64
actor_critic_model = ActorCritic(input_shape, action_size)


# Experience Replay Buffer
replay_buffer = init_replay_buffer()

def get_legal_move_mask(board):
    legal_mask = np.zeros((64, 64), dtype=np.float32)
    for move in board.legal_moves:
        legal_mask[move.from_square, move.to_square] = 1.0
    return np.expand_dims(legal_mask, axis=(0, -1))

def encode_history_as_tensor(move_history):
    """Converts move history into a tensor format directly (skip vocab)."""
    history_tensor = np.zeros((50, 4), dtype=np.float32)  # 50 moves max, each move is (from, to) = (x1, y1, x2, y2)

    for idx, move in enumerate(move_history[-50:]):
        from_sq, to_sq = chess.parse_square(move[:2]), chess.parse_square(move[2:])
        from_x, from_y = divmod(from_sq, 8)
        to_x, to_y = divmod(to_sq, 8)
        history_tensor[idx] = [from_x, from_y, to_x, to_y]

    return history_tensor


def play_vs_stockfish(skill_level, athena_is_white, epoch):
    """Athena plays against Stockfish with increasing difficulty."""
    board = chess.Board()
    move_history = []
    move_count = 0 

    apollo_discount_factor = 1.0
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
        stockfish.configure({"Skill Level": skill_level})  

        while not board.is_game_over():
            move_count += 1
            fen_before = board.fen()
            attack_map, defense_map = get_attack_defense_maps(board)

            fen_tensor = np.expand_dims(fen_to_tensor(fen_before), axis=0)
            move_history_encoded = encode_history_as_tensor(move_history)
            move_history_encoded = np.expand_dims(move_history_encoded, axis=0)  # Batchify it
            attack_map = np.expand_dims(attack_map, axis=(0, -1))
            defense_map = np.expand_dims(defense_map, axis=(0, -1))
            turn_indicator = np.array([[1.0 if board.turn == chess.WHITE else 0.0]], dtype=np.float32)
            legal_moves_mask = get_legal_move_mask(board)

            state = (fen_tensor, move_history_encoded, attack_map, defense_map, turn_indicator, legal_moves_mask)

            # Ensure legal_moves is a list of chess.Move objects
            legal_moves_list = list(board.legal_moves)

            if (board.turn == chess.WHITE and athena_is_white) or (board.turn == chess.BLACK and not athena_is_white):
                action = None  # Default action to None

                # Apollo: Opening Book Moves (First 10 Moves)
                if len(move_history) < 20 and random.random() < apollo_discount_factor:
                    apollo_move = apollo.get_opening_move(board)
                    if apollo_move and apollo_move in legal_moves_list:
                        action = apollo_move
                        print(f"Apollo move: {action}")
                    else:
                        print("❌ No book move found in Apollo.")

                # Hades: Endgame (≤7 Pieces)
                elif len(board.piece_map()) <= 7 and random.random() < 0.8:  # 80% chance to use Hades
                    hades_move = hades.get_best_endgame_move(board)
                    if hades_move and hades_move in legal_moves_list:
                        action = hades_move
                    else:
                        print("❌ No endgame move found in Hades.")

                # Game Memory Recall (Pattern Matching)
                if action is None:
                    past_game = replay_buffer.game_memory.find_similar_position(board)
                    if past_game:
                        print("♟️ Found similar position in memory. Recalling pattern.")
                        suggested_move = chess.Move.from_uci(past_game["moves"][0])
                        if suggested_move in legal_moves_list:
                            action = suggested_move

                # RL + 20% Exploration
                if action is None:
                    if random.random() < 0.8:
                        action = actor_critic_model.sample_action(state, legal_moves_list)  # 80% RL move
                    else:
                        action = random.choice(legal_moves_list)  # 20% Random Exploration
                apollo_discount_factor *= 0.95
                
                # Final Check Before Execution
                if action not in legal_moves_list:
                    print(f"⚠️ Athena attempted illegal move: {action}. Selecting a random legal move.")
                    action = random.choice(legal_moves_list)

            # Stockfish Plays If Not Athena's Turn
            else:
                stockfish_move = stockfish.play(board, chess.engine.Limit(time=0.1)).move
                if stockfish_move in legal_moves_list:
                    action = stockfish_move
                else:
                    print("⚠️ Stockfish tried an illegal move. Selecting a random legal move.")
                    action = random.choice(legal_moves_list)  # Fallback safety

   

            # Execute Move
            board_before = board.copy()
            board.push(action)  # No more illegal moves!
            move_history.append(action.uci())


            fen_after = board.fen()
            fen_tensor_after = np.expand_dims(fen_to_tensor(fen_after), axis=0)
            legal_moves_mask = get_legal_move_mask(board)

            # Determine the game result from Athena's perspective
            game_result = None
            if board.is_game_over():
                outcome = board.outcome()
                if outcome.winner is None:
                    game_result = 0  # Draw
                elif outcome.winner == chess.WHITE:  # White wins
                    if athena_is_white:
                        game_result = 1  # Athena (White) wins
                    else:
                        game_result = -1  # Athena (Black) loses
                else:
                    if athena_is_white:
                        game_result = -1  # Athena (White) loses
                    else:
                        game_result = 1  # Athena (Black) wins

            reward = compute_reward(board_before, board, game_result)
            next_state = (fen_tensor_after, move_history_encoded, attack_map, defense_map, turn_indicator, legal_moves_mask)

            encoded_move = action.uci()
            replay_buffer.add(state, encoded_move, reward, next_state, board.is_game_over(), fen_after)
            print(f"Move {move_count}: {action.uci()} ({'Athena' if (board.turn != athena_is_white) else 'Stockfish'})")
            logging.info(f"Move {move_count}: {action.uci()} ({'Athena' if (board.turn != athena_is_white) else 'Stockfish'})")

    print(f"Game Over. Result: {board.result()}")
    logging.info(f"Game Over. Result: {board.result()}")


def train_athena():
    """Train Athena using PPO with graph-based encoding."""
    print("\n Training Athena Started...\n")
    logging.info("Training Athena Started...")
    
    for epoch in range(EPOCHS):
        progress = epoch / EPOCHS
        if progress < 0.25:  # 0% - 25%
            skill_level = random.randint(0, 3)  # Beginner
        elif progress < 0.50:  # 25% - 50%
            skill_level = random.randint(4, 7)  # Intermediate
        elif progress < 0.75:  # 50% - 75%
            skill_level = random.randint(8, 12)  # Advanced
        elif progress < 0.90:  # 75% - 90%
            skill_level = random.randint(13, 16)  # Stronger
        else:  # Last 10%
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
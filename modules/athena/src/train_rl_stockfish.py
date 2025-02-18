import tensorflow as tf
import numpy as np
import chess
import chess.engine
import random
import json
from datetime import datetime
from model import get_model
from reward_function import compute_reward
from dqn import DQN, train_dqn
from replay_buffer import init_replay_buffer
from utils import fen_to_tensor, move_to_index, build_move_vocab
import os
if os.path.exists("replay_buffer.h5"):
    os.remove("replay_buffer.h5")
    print("🚀 Corrupt replay buffer deleted! Restart training.")


# Load move vocabulary
MAX_VOCAB_SIZE = 500000

# move_vocab = build_move_vocab("move_vocab.json")
with open("move_vocab.json", "r") as f:
    move_vocab = json.load(f)


# RL Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99  # Discount factor for rewards
EPSILON = 0.1  # Exploration rate for epsilon-greedy policy
BATCH_SIZE = 32
EPOCHS = 200

# Paths
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = f"models/athena_DQN_{timestamp}_{EPOCHS}epochs.keras"

# Load Athena Model
input_shape = (8, 8, 20)
action_size = 64 * 64
dqn_model = DQN(input_shape, action_size)

# Experience Replay Buffer
replay_buffer = init_replay_buffer()

# Adaptive Move Vocabulary Management
def update_move_vocab(move):
    """Dynamically updates the move vocabulary, forgetting less useful moves."""
    if move not in move_vocab:
        if len(move_vocab) >= MAX_VOCAB_SIZE:
            least_used_move = min(move_vocab, key=move_vocab.get)
            del move_vocab[least_used_move]  # Remove least useful move
        move_vocab[move] = 1  # Add new move
    else:
        move_vocab[move] += 1  # Increase usage count

    # Save updated move vocabulary
    with open("move_vocab.json", "w") as f:
        json.dump(move_vocab, f, indent=4)

def play_vs_stockfish(skill_level, athena_is_white):
    """Athena plays against Stockfish with increasing difficulty."""
    board = chess.Board()
    move_history = []

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
        stockfish.configure({"Skill Level": skill_level})  # Set Stockfish strength
        
        while not board.is_game_over():
            fen_before = board.fen()
            legal_moves = list(board.legal_moves)

            legal_moves_mask = np.zeros((64, 64), dtype=np.int8)
            for legal_move in board.legal_moves:
                legal_moves_mask[legal_move.from_square, legal_move.to_square] = 1

            fen_tensor = np.expand_dims(fen_to_tensor(fen_before), axis=0)
            move_history_encoded = np.array(move_to_index(move_history, move_vocab), dtype=np.int32).reshape(1, 50)
            legal_moves_mask = np.expand_dims(legal_moves_mask, axis=0)
            turn_indicator = np.array([[1.0 if board.turn == chess.WHITE else 0.0]], dtype=np.float32)

            if (board.turn == chess.WHITE and athena_is_white) or (board.turn == chess.BLACK and not athena_is_white):
                # Athena plays
                q_values = dqn_model([fen_tensor, move_history_encoded, legal_moves_mask, turn_indicator])
                move_index = np.argmax(q_values.numpy()[0])
                athena_move = legal_moves[move_index] if move_index < len(legal_moves) else random.choice(legal_moves)
            else:
                # Stockfish plays
                stockfish_move = stockfish.play(board, chess.engine.Limit(time=0.1))
                athena_move = stockfish_move.move

            board_before = board.copy()
            board.push(athena_move)
            move_history.append(athena_move.uci())

            game_result = None
            if board.is_game_over():
                outcome = board.outcome()
                if outcome.winner is None:  # Draw
                    game_result = 0
                elif outcome.winner == chess.WHITE:
                    game_result = 1
                else:
                    game_result = -1

            reward = compute_reward(board_before, board, athena_move, game_result)
            next_fen = board.fen()
            next_legal_moves_mask = np.zeros((64, 64), dtype=np.int8)
            for legal_move in board.legal_moves:
                next_legal_moves_mask[legal_move.from_square, legal_move.to_square] = 1
            next_legal_moves_mask = np.expand_dims(next_legal_moves_mask, axis=0)

            next_state = (next_fen, move_history_encoded, next_legal_moves_mask, turn_indicator)

            replay_buffer.add((fen_before, move_history_encoded, legal_moves_mask, turn_indicator),
                              athena_move.uci(), reward, next_state, board.is_game_over())

def train_athena():
    """Train Athena by playing against Stockfish."""
    for epoch in range(EPOCHS):
        # Adaptive difficulty range per epoch block
        if epoch < 50:
            skill_level = random.randint(0, 3)
        elif epoch < 100:
            skill_level = random.randint(4, 7)
        elif epoch < 150:
            skill_level = random.randint(8, 11)
        elif epoch < 175:
            skill_level = random.randint(12, 15)
        else:
            skill_level = random.randint(16, 20)

        # Assign random color to Athena
        athena_is_white = bool(random.getrandbits(1))  
        white_player = "Athena" if athena_is_white else "Stockfish"

        print(f"\n🌟 Epoch {epoch + 1}/{EPOCHS} | Stockfish Skill Level: {skill_level} | White: {white_player}")

        # Play a game against Stockfish
        play_vs_stockfish(skill_level, athena_is_white)
        
        # Train DQN
        print(len(replay_buffer.buffer))
        all_transitions = replay_buffer  # Fetch all moves
        loss = train_dqn(dqn_model, all_transitions, GAMMA)  # Train on the full game
        
        if loss is not None:
            print(f"🎯 Training Step | Epoch {epoch + 1} | Loss: {np.mean(loss):.4f}")
        else:
            print("⚠️ Skipping training step - Not enough data in replay buffer.")

    replay_buffer.save()
    dqn_model.save(MODEL_SAVE_PATH)
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")



if __name__ == "__main__":
    train_athena()

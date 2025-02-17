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
EPOCHS = 5

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


def play_game():
    """Plays a self-play game between Athena and itself while collecting training data."""
    board = chess.Board()
    move_history = []
    
    while not board.is_game_over():
        fen_before = board.fen()  # Store raw FEN string
        legal_moves = list(board.legal_moves)

        legal_moves_mask = np.zeros((64, 64), dtype=np.int8)
        for legal_move in board.legal_moves:
            legal_moves_mask[legal_move.from_square, legal_move.to_square] = 1

        # Convert to tensor for model input
        fen_tensor = np.expand_dims(fen_to_tensor(fen_before), axis=0)
        move_history_encoded = np.array(move_to_index(move_history, move_vocab), dtype=np.int32).reshape(1, 50)
        legal_moves_mask = np.expand_dims(legal_moves_mask, axis=0)
        turn_indicator = np.array([[1.0 if board.turn == chess.WHITE else 0.0]], dtype=np.float32)

        # Get Athena's move from DQN
        q_values = dqn_model([fen_tensor, move_history_encoded, legal_moves_mask, turn_indicator])

        if random.random() < EPSILON:
            athena_move = random.choice(legal_moves)  # Exploration
        else:
            move_index = np.argmax(q_values.numpy()[0])
            athena_move = legal_moves[move_index] if move_index < len(legal_moves) else random.choice(legal_moves)

        board_before = board.copy()
        board.push(athena_move)
        move_history.append(athena_move.uci())

        game_result = None  # Default to neutral if the game is ongoing

        if board.is_game_over():
            outcome = board.outcome()
            if outcome.winner is None:  # Draw (stalemate or repetition)
                game_result = 0
            elif outcome.winner == chess.WHITE:
                game_result = 1  # White won
            else:
                game_result = -1  # Black won

        # Compute reward based on piece-based shaping
        reward = compute_reward(board_before, board, athena_move, game_result)

        # Generate the next state
        next_fen = board.fen()
        next_legal_moves_mask = np.zeros((64, 64), dtype=np.int8)
        for legal_move in board.legal_moves:
            next_legal_moves_mask[legal_move.from_square, legal_move.to_square] = 1
        next_legal_moves_mask = np.expand_dims(next_legal_moves_mask, axis=0)

        # Store FEN string instead of tensor
        next_state = (next_fen, move_history_encoded, next_legal_moves_mask, turn_indicator)

        replay_buffer.add((fen_before, move_history_encoded, legal_moves_mask, turn_indicator),
                          athena_move.uci(), reward, next_state, board.is_game_over())



def train_athena():
    """Trains Athena using collected self-play data with reinforcement learning."""
    for epoch in range(EPOCHS):
        print(f"\n🌟 Epoch {epoch + 1}/{EPOCHS}")

        # Play a game and collect data
        play_game()
        
        # Train DQN with experience replay
        loss = train_dqn(dqn_model, replay_buffer, BATCH_SIZE, GAMMA)
        
        # Training Info 📊
        if loss is not None:
            print(f"🎯 Training Step | Epoch {epoch + 1} | Loss: {np.mean(loss):.4f}")
        else:
            print("⚠️ Skipping training step - Not enough data in replay buffer.")

    replay_buffer.save()
    dqn_model.save(MODEL_SAVE_PATH)
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_athena()

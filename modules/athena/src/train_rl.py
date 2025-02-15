import tensorflow as tf
import numpy as np
import chess
import chess.engine
import random
import json
from model import get_model
from reward_function import compute_reward
from dqn import DQN, train_dqn
from replay_buffer import init_replay_buffer
from utils import fen_to_tensor, move_to_index, build_move_vocab

# Load move vocabulary
MAX_VOCAB_SIZE = 500000

# move_vocab = build_move_vocab("move_vocab.json")
with open("move_vocab.json", "r") as f:
    move_vocab = json.load(f)

# Paths
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
MODEL_SAVE_PATH = "models/athena_gm_trained_20250211_090344_10epochs.keras"

# RL Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99  # Discount factor for rewards
EPSILON = 0.1  # Exploration rate for epsilon-greedy policy
BATCH_SIZE = 32
EPOCHS = 5

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
    """Plays a game between Athena and Stockfish while collecting training data."""
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    move_history = []

    move_history_encoded = np.zeros((50,), dtype=np.int32)  # Initialize empty move history
    legal_moves_mask = np.zeros((64, 64), dtype=np.int8)  # Initialize legal moves mask

    while not board.is_game_over():
        fen = board.fen()
        turn_indicator = np.array([1 if board.turn == chess.WHITE else 0], dtype=np.float32)

        # Update legal move mask for the current position
        legal_moves_mask = np.zeros((64, 64), dtype=np.int8)
        for legal_move in board.legal_moves:
            legal_moves_mask[legal_move.from_square, legal_move.to_square] = 1

        # Ensure eval_before is always valid
        try:
            eval_before = engine.analyse(board, chess.engine.Limit(depth=10))["score"].relative.score(mate_score=10000) / 100.0
        except Exception:
            eval_before = 0.0

        # Ensure move history is initialized correctly
        fen_tensor = np.expand_dims(fen_to_tensor(fen), axis=0)
        move_history_encoded = np.array(move_to_index(move_history, move_vocab), dtype=np.int32).reshape(1, 50)  # Correct Shape (1, 50)
        legal_moves_mask = np.expand_dims(legal_moves_mask, axis=0)  # Expands to (1, 64, 64)
        turn_indicator = np.expand_dims(np.array([turn_indicator], dtype=np.float32), axis=0)  # Expands to (1, 1)
        eval_before = np.expand_dims(np.array([eval_before], dtype=np.float32), axis=0)  # Expands to (1, 1)

        # Athena Move Selection
        q_values = dqn_model([fen_tensor, move_history_encoded, legal_moves_mask, eval_before, turn_indicator])

        if random.random() < EPSILON:
            athena_move = random.choice(list(board.legal_moves))  # Explore
        else:
            legal_moves = list(board.legal_moves)  # Convert the generator to a list
            move_index = np.argmax(q_values.numpy()[0])
            athena_move = legal_moves[move_index] if move_index < len(legal_moves) else random.choice(legal_moves)

        update_move_vocab(athena_move.uci())
        board.push(athena_move)
        move_history.append(athena_move.uci())  # Store move history

    engine.quit()



def train_athena():
    """Trains Athena using collected self-play data with reinforcement learning."""
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        play_game()
        
        # Train DQN with experience replay
        loss = train_dqn(dqn_model, replay_buffer, BATCH_SIZE, GAMMA)
        if loss is not None:
            print(f"Loss: {loss:.4f}")  # Only print if loss is valid
        else:
            print("Skipping training step - Not enough data in replay buffer.")  # Debug message
    
    replay_buffer.save()
    dqn_model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_athena()

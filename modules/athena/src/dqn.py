import os
import json
import tensorflow as tf
import numpy as np
from utils import fen_to_tensor, move_to_index, build_move_vocab
from tensorflow.keras import layers, Model
from model import TransformerBlock  # Use the Transformer block from original model

MAX_VOCAB_SIZE = 500000
TARGET_UPDATE_FREQUENCY = 500  # Update target network every N steps
TAU = 0.05  # Soft update factor for target network

class DQN(tf.keras.Model):
    def __init__(self, input_shape, action_size, learning_rate=1e-4):
        super(DQN, self).__init__()

        # Inputs
        board_input = layers.Input(shape=(8, 8, 20))  # FEN tensor
        move_history_input = layers.Input(shape=(50,), dtype=tf.int32)  # Move history
        legal_moves_input = layers.Input(shape=(64, 64), dtype=tf.float32)  # Legal move mask
        turn_input = layers.Input(shape=(1,), dtype=tf.float32)  # Turn indicator

        # Board Processing (CNN)
        x = layers.Conv2D(128, 3, activation='gelu', padding='same')(board_input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, activation='gelu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Reshape((64, 256))(x)  
        x = layers.GlobalAvgPool1D()(x)  

        # Move History Encoding (Transformer)
        emb = layers.Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=128)(move_history_input)
        y = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512, rate=0.3)(emb)
        y = layers.GlobalAvgPool1D()(y)

        # Fusion
        turn_scaled = layers.Dense(32, activation="gelu")(turn_input)
        fused = layers.Concatenate()([x, y, turn_scaled])

        z = layers.Dense(512, activation='gelu')(fused)
        z = layers.Dropout(0.3)(z)

        # Move Selection Output
        move_output = layers.Dense(6, activation='linear', name="move_output")(z)

        # Define Model
        self.model = Model(inputs=[board_input, move_history_input, legal_moves_input, turn_input], 
                                    outputs=move_output)

        # Target Network (Copy of Main Network)
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        # Compile with AdamW Optimizer
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
            loss="mse"
        )

    def call(self, inputs):
        return self.model(inputs)

    def update_target_network(self):
        """Soft update target network weights using TAU (Polyak averaging)."""
        model_weights = np.array(self.model.get_weights(), dtype=object)
        target_weights = np.array(self.target_model.get_weights(), dtype=object)
        new_weights = TAU * model_weights + (1 - TAU) * target_weights
        self.target_model.set_weights(new_weights)

    def predict_value(self, fen_tensor):
        """Predicts board evaluation (replaces Stockfish eval)."""
        _, board_eval = self.model([fen_tensor, np.zeros((1, 50)), np.zeros((1, 64, 64)), np.zeros((1, 1)), np.zeros((1, 1))])
        return board_eval.numpy()[0, 0]


# Load move vocabulary
VOCAB_FILE = "move_vocab.json"

if os.path.exists(VOCAB_FILE):
    with open(VOCAB_FILE, "r") as f:
        move_vocab = json.load(f)
else:
    print("🚨 No move vocabulary found! Creating a new one...")
    move_vocab = {}  # Start empty


def train_dqn(dqn_model, replay_buffer, batch_size=32, gamma=0.99, training_step=0):
    """Trains the DQN model using experience replay."""
    if replay_buffer.size() < batch_size:
        return None  # Return None if not enough samples

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Ensure next_states are not None before unpacking
    valid_indices = [i for i, ns in enumerate(next_states) if ns is not None]

    if not valid_indices:
        print("⚠️ No valid next states in replay buffer. Skipping training step.")
        return None

    states = [states[i] for i in valid_indices]
    actions = [actions[i] for i in valid_indices]
    rewards = [rewards[i] for i in valid_indices]
    next_states = [next_states[i] for i in valid_indices]
    dones = np.array([dones[i] for i in valid_indices], dtype=np.float32)  # Convert to NumPy array

    # Unpack values after filtering
    state_fens, state_histories, state_legal_moves, state_turn = zip(*states)
    next_fens, next_histories, next_legal_moves, next_turn = zip(*next_states)

    # Convert FEN strings to tensors only if they are strings
    states = [fen_to_tensor(fen) if isinstance(fen, str) else fen for fen in state_fens]
    next_states = [fen_to_tensor(fen) if isinstance(fen, str) else fen for fen in next_fens]

    # Fix Move History Shape Issue
    state_histories = np.array(state_histories, dtype=np.int32).squeeze(axis=1)  
    next_histories = np.array(next_histories, dtype=np.int32).squeeze(axis=1)  

    # Fix Legal Moves Shape Issue
    state_legal_moves = np.array(state_legal_moves, dtype=np.float32).squeeze(axis=1)  
    next_legal_moves = np.array(next_legal_moves, dtype=np.float32).squeeze(axis=1)  

    # Convert moves into integer indices
    action_indices = np.array([[i, move_to_index(move, move_vocab)[0]] for i, move in enumerate(actions)], dtype=np.int32)

    # Use Target Network for Q-value estimation
    target_q_values = dqn_model.target_model([
        np.array(next_states, dtype=np.float32),
        next_histories,
        next_legal_moves,
        np.array(next_turn, dtype=np.float32)
    ])

    max_q_values = np.max(target_q_values.numpy(), axis=1)
    target_values = np.reshape(rewards + gamma * max_q_values * (1 - dones), (-1, 1))  # Fixed issue

    with tf.GradientTape() as tape:
        q_values = dqn_model([
            np.array(states, dtype=np.float32),
            state_histories,
            state_legal_moves,
            np.array(state_turn, dtype=np.float32)
        ])

        q_values_selected = tf.gather_nd(q_values, action_indices)
        loss = tf.keras.losses.MSE(target_values, q_values_selected)

    grads = tape.gradient(loss, dqn_model.model.trainable_variables)
    dqn_model.model.optimizer.apply_gradients(zip(grads, dqn_model.model.trainable_variables))

    # Update target network periodically
    if training_step % TARGET_UPDATE_FREQUENCY == 0:
        dqn_model.update_target_network()

    return loss.numpy()


if __name__ == "__main__":
    # Example usage
    input_shape = (8, 8, 20)  # Assuming 20-layer input for board representation
    action_size = 64 * 64  # All possible chess moves
    dqn_model = DQN(input_shape, action_size)
    print("DQN model initialized!")

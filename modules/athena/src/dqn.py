import os
import json
import tensorflow as tf
import numpy as np
from utils import fen_to_tensor, move_to_index, build_move_vocab
from tensorflow.keras import layers, Model
from model import TransformerBlock  # Use the Transformer block from original model

MAX_VOCAB_SIZE = 500000

class DQN(tf.keras.Model):
    def __init__(self, input_shape, action_size, learning_rate=1e-4):
        super(DQN, self).__init__()

        # Inputs
        board_input = tf.keras.layers.Input(shape=(8, 8, 20))  # FEN tensor
        move_history_input = tf.keras.layers.Input(shape=(50,), dtype=tf.int32)  # Move history
        legal_moves_input = tf.keras.layers.Input(shape=(64, 64), dtype=tf.float32)  # Legal move mask
        turn_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)  # Turn indicator

        # Board Processing (CNN)
        x = tf.keras.layers.Conv2D(128, 3, activation='gelu', padding='same')(board_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(256, 3, activation='gelu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Reshape((64, 256))(x)  
        x = tf.keras.layers.GlobalAvgPool1D()(x)  

        # Move History Encoding (Transformer)
        emb = tf.keras.layers.Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=128)(move_history_input)
        y = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512, rate=0.3)(emb)
        y = tf.keras.layers.GlobalAvgPool1D()(y)

        # Fusion
        turn_scaled = tf.keras.layers.Dense(32, activation="gelu")(turn_input)
        fused = tf.keras.layers.Concatenate()([x, y, turn_scaled])

        z = tf.keras.layers.Dense(512, activation='gelu')(fused)
        z = tf.keras.layers.Dropout(0.3)(z)

        # Move Selection Output
        move_output = tf.keras.layers.Dense(6, activation='linear', name="move_output")(z)

        # Define Model
        self.model = tf.keras.Model(inputs=[board_input, move_history_input, legal_moves_input, turn_input], 
                                    outputs=move_output)


        # Compile with AdamW Optimizer
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
            loss="mse"
        )

    def call(self, inputs):
        return self.model(inputs)
    
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


def train_dqn(dqn_model, replay_buffer, batch_size=32, gamma=0.99):
    """Trains the DQN model using experience replay."""
    if replay_buffer.size() < batch_size:
        return None  # Return None if not enough samples

    minibatch = list(zip(*replay_buffer.sample(batch_size)))  # Unzip correctly

    state_fens, state_histories, state_legal_moves, state_eval, state_turn = zip(*minibatch[0])
    actions, rewards, next_state_data, dones = minibatch[1], minibatch[2], minibatch[3], minibatch[4]
    next_fens, next_histories, next_legal_moves, next_eval, next_turn = zip(*next_state_data)



    # Convert FEN strings to tensors
    states = np.array([fen_to_tensor(fen) for fen in states], dtype=np.float32)
    next_states = np.array([fen_to_tensor(fen) for fen in next_states], dtype=np.float32)

    # Convert UCI moves into integer indices using `move_vocab`
    action_indices = np.array([[i, move_to_index([move], move_vocab, max_sequence_length=1)[0]] for i, move in enumerate(actions)], dtype=np.int32)

    target_q_values = dqn_model([
                                    np.array(next_fens, dtype=np.float32),
                                    np.array(next_histories, dtype=np.int32),
                                    np.array(next_legal_moves, dtype=np.float32),
                                    np.array(next_eval, dtype=np.float32),
                                    np.array(next_turn, dtype=np.float32)
                                ])

    max_q_values = np.max(target_q_values.numpy(), axis=1)
    target_values = np.reshape(rewards + gamma * max_q_values * (1 - dones), (-1, 1))

    with tf.GradientTape() as tape:
        q_values = dqn_model([
            np.array(state_fens, dtype=np.float32),
            np.array(state_histories, dtype=np.int32),
            np.array(state_legal_moves, dtype=np.float32),
            np.array(state_eval, dtype=np.float32),
            np.array(state_turn, dtype=np.float32)
        ])

        q_values_selected = tf.gather_nd(q_values, action_indices)
        loss = tf.keras.losses.MSE(target_values, q_values_selected)
    
    grads = tape.gradient(loss, dqn_model.trainable_variables)
    dqn_model.optimizer.apply_gradients(zip(grads, dqn_model.trainable_variables))

    return loss.numpy()



if __name__ == "__main__":
    # Example usage
    input_shape = (8, 8, 20)  # Assuming 20-layer input for board representation
    action_size = 64 * 64  # All possible chess moves
    dqn_model = DQN(input_shape, action_size)
    print("DQN model initialized!")

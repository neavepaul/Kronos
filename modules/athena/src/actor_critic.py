import tensorflow as tf
import numpy as np
import random
import chess
import chess.engine
from tensorflow.keras import layers, Model
import sys
from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parents[3]  # Moves up to "Kronos/"
sys.path.append(str(ROOT_PATH))

from modules.athena.src.utils import fen_to_tensor, move_to_index, encode_move, index_to_move
from modules.athena.src.transformer_board_encoder import TransformerBoardEncoder

# PPO Hyperparameters
GAMMA = 0.99        # Discount factor for future rewards
LAMBDA = 0.95       # GAE smoothing
EPSILON = 0.2       # PPO clipping
LR = 3e-4           # Learning rate

MAX_VOCAB_SIZE = 500000  # Move history size
ENTROPY_BONUS = 0.01  # Encourages exploration


class ActorCritic(Model):
    """
    Actor-Critic (PPO-style) Chess Model:
    - Actor predicts move probabilities.
    - Critic evaluates the board position.
    """

    def __init__(self, input_shape, action_size):
        super(ActorCritic, self).__init__()

        # Inputs
        board_input = layers.Input(shape=(8, 8, 20))  # FEN tensor
        move_history_input = layers.Input(shape=(50,), dtype=tf.int32)  # Move history
        attack_map_input = layers.Input(shape=(8, 8, 1), dtype=tf.float32)  # Attack Map
        defense_map_input = layers.Input(shape=(8, 8, 1), dtype=tf.float32)  # Defense Map
        turn_input = layers.Input(shape=(1,), dtype=tf.float32)  # Turn indicator

        # Board Processing (Transformer-based Encoding)
        self.board_encoder = TransformerBoardEncoder(embed_dim=128, num_heads=4, ff_dim=512, num_layers=4)
        board_features = self.board_encoder(board_input)

        # Move History Encoding (Transformer)
        emb = layers.Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=128)(move_history_input)
        move_history_features = layers.LSTM(128, return_sequences=True)(emb)
        move_history_features = layers.GlobalAvgPool1D()(move_history_features)

        # Attack & Defense Maps Processing
        attack_features = layers.Conv2D(64, (3, 3), activation='gelu', padding='same')(attack_map_input)
        defense_features = layers.Conv2D(64, (3, 3), activation='gelu', padding='same')(defense_map_input)
        attack_features = layers.GlobalAvgPool2D()(attack_features)
        defense_features = layers.GlobalAvgPool2D()(defense_features)

        # Fusion
        turn_scaled = layers.Dense(32, activation="gelu")(turn_input)
        fused = layers.Concatenate()([board_features, move_history_features, attack_features, defense_features, turn_scaled])

        # Shared Layers
        z = layers.Dense(512, activation='gelu')(fused)
        z = layers.Dropout(0.3)(z)

        # **Actor (Policy Network)**
        policy_logits = layers.Dense(action_size, activation='linear', name="policy_logits")(z)

        # **Critic (Value Network)**
        value_output = layers.Dense(1, activation='linear', name="value_output")(z)

        # Define Model
        self.model = Model(
            inputs=[board_input, move_history_input, attack_map_input, defense_map_input, turn_input],
            outputs=[policy_logits, value_output]
        )

        # Compile with AdamW Optimizer
        self.model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LR), loss=["categorical_crossentropy", "mse"])

    def call(self, inputs):
        return self.model(inputs)

    def get_policy_value(self, inputs):
        """Returns the policy logits and value estimate for a given state."""
        return self.model(inputs)

    def sample_action(self, inputs, legal_moves):
        """Samples an action using the policy network and ensures only legal moves are chosen."""
        policy_logits, _ = self.get_policy_value(inputs)

        # Convert legal moves to UCI format
        legal_moves_list = list(legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves_list]

        # Generate a probability distribution for legal moves only
        move_probs = tf.nn.softmax(policy_logits)

        # Select a move **only from legal moves**
        move_index = tf.random.categorical(tf.math.log(move_probs), num_samples=1).numpy()[0, 0]

        if move_index < len(legal_moves_list):
            return legal_moves_list[move_index]  # ✅ Guaranteed legal move

        return random.choice(legal_moves_list)  # Fallback safety

def compute_advantage(rewards, values, gamma=GAMMA, lambda_=LAMBDA):
    """Computes GAE (Generalized Advantage Estimation) for PPO updates."""
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    advantages = np.zeros_like(deltas)
    acc = 0
    for t in reversed(range(len(deltas))):
        acc = deltas[t] + gamma * lambda_ * acc
        advantages[t] = acc
    return advantages


def train_actor_critic(actor_critic_model, replay_buffer, batch_size=32):
    """Trains the Actor-Critic model using PPO updates."""
    if len(replay_buffer.buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Compute value estimates
    _, values = actor_critic_model.get_policy_value(states)
    _, next_values = actor_critic_model.get_policy_value(next_states)

    # Compute advantages
    advantages = compute_advantage(rewards, values)

    with tf.GradientTape() as tape:
        policy_logits, value_preds = actor_critic_model.get_policy_value(states)

        # Compute policy loss with PPO clipping
        action_probs = tf.nn.softmax(policy_logits)
        selected_action_probs = tf.gather_nd(action_probs, actions)
        old_action_probs = tf.stop_gradient(selected_action_probs)
        ratio = selected_action_probs / (old_action_probs + 1e-8)

        # Clipped surrogate loss
        clipped_ratio = tf.clip_by_value(ratio, 1 - EPSILON, 1 + EPSILON)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        # Value loss
        value_loss = tf.reduce_mean(tf.square(value_preds - rewards))

        # Entropy Bonus (Encourages Exploration)
        entropy = -tf.reduce_mean(action_probs * tf.math.log(action_probs + 1e-8))
        total_loss = policy_loss + 0.5 * value_loss - ENTROPY_BONUS * entropy

    # Apply gradients
    grads = tape.gradient(total_loss, actor_critic_model.model.trainable_variables)
    actor_critic_model.model.optimizer.apply_gradients(zip(grads, actor_critic_model.model.trainable_variables))

    return total_loss.numpy()

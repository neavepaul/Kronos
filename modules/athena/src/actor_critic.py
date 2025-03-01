import chess
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model

from modules.athena.src.transformer_board_encoder import TransformerBoardEncoder

# PPO Hyperparameters
GAMMA = 0.99        # Discount factor for future rewards
LAMBDA = 0.95       # GAE smoothing
EPSILON = 0.2       # PPO clipping
LR = 1e-4           # Learning rate
ENTROPY_BONUS = 0.01  # Exploration term


class ActorCritic(Model):
    """
    Actor-Critic (PPO) Model with Built-in Legal Move Masking.
    Policy head outputs masked probabilities, legal mask is baked into model graph.
    """

    def __init__(self, input_shape, action_size):
        super(ActorCritic, self).__init__()

        # Inputs
        board_input = layers.Input(shape=(8, 8, 20))  # FEN tensor
        move_history_input = layers.Input(shape=(50, 4), dtype=tf.float32)  # Last 50 moves (x1, y1, x2, y2)
        attack_map_input = layers.Input(shape=(8, 8, 1), dtype=tf.float32)  # Attack Map
        defense_map_input = layers.Input(shape=(8, 8, 1), dtype=tf.float32)  # Defense Map
        turn_input = layers.Input(shape=(1,), dtype=tf.float32)  # Turn indicator (0=black, 1=white)
        legal_moves_mask_input = layers.Input(shape=(64, 64, 1), dtype=tf.float32)  # Legal moves mask

        # Board Processing (Transformer)
        self.board_encoder = TransformerBoardEncoder(embed_dim=128, num_heads=4, ff_dim=512, num_layers=4)
        board_features = self.board_encoder(board_input)

        # Move History Processing (Coordinate MLP)
        move_history_flat = layers.Flatten()(move_history_input)
        move_history_features = layers.Dense(128, activation='gelu')(move_history_flat)
        move_history_features = layers.Dense(64, activation='gelu')(move_history_features)

        # Attack/Defense Maps
        attack_features = layers.Conv2D(64, (3, 3), activation='gelu', padding='same')(attack_map_input)
        defense_features = layers.Conv2D(64, (3, 3), activation='gelu', padding='same')(defense_map_input)
        attack_features = layers.GlobalAvgPool2D()(attack_features)
        defense_features = layers.GlobalAvgPool2D()(defense_features)

        # Legal Move Mask Processing
        legal_features = layers.Conv2D(32, (3, 3), activation='gelu', padding='same')(legal_moves_mask_input)
        legal_features = layers.GlobalAvgPool2D()(legal_features)

        # Turn Encoding
        turn_scaled = layers.Dense(32, activation="gelu")(turn_input)

        # Fusion
        fused = layers.Concatenate()([
            board_features, move_history_features, attack_features, defense_features, turn_scaled, legal_features
        ])

        # Shared Representation
        z = layers.Dense(512, activation='gelu')(fused)
        z = layers.Dropout(0.3)(z)

        # Actor Head (Policy)
        policy_logits = layers.Dense(64 * 64, activation='linear', name="policy_logits")(z)

        # Legal Move Mask Flatten (64x64 -> 1x4096)
        legal_mask_flat = layers.Flatten()(legal_moves_mask_input)

        # Masking - Apply mask before softmax inside the model
        masked_logits = layers.Lambda(lambda x: x[0] + tf.math.log(x[1] + 1e-8))([policy_logits, legal_mask_flat])

        # Final Policy (Softmax after masking)
        policy_output = layers.Softmax(name="policy_output")(masked_logits)

        # Critic Head (Value)
        value_output = layers.Dense(1, activation='linear', name="value_output")(z)

        # Define Full Model
        self.model = Model(
            inputs=[board_input, move_history_input, attack_map_input, defense_map_input, turn_input, legal_moves_mask_input],
            outputs=[policy_output, value_output]
        )

        self.model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LR), loss=["categorical_crossentropy", "mse"])

    def call(self, inputs):
        return self.model(inputs)

    def get_policy_value(self, inputs):
        return self.model(inputs)

    def sample_action(self, inputs, legal_moves_list):
        policy_probs, _ = self.get_policy_value(inputs)

        # Mask out illegal moves
        legal_moves_indices = [self.uci_to_index(move) for move in legal_moves_list]
        legal_policy_probs = np.zeros_like(policy_probs.numpy().ravel())
        legal_policy_probs[legal_moves_indices] = policy_probs.numpy().ravel()[legal_moves_indices]

        # Normalize the probabilities
        legal_policy_probs /= legal_policy_probs.sum()

        # Sample move directly from legal-only policy distribution
        move_index = np.random.choice(64 * 64, p=legal_policy_probs)

        from_square = move_index // 64
        to_square = move_index % 64
        return from_square, to_square

    def uci_to_index(self, move):
        from_sq = move.from_square
        to_sq = move.to_square
        return from_sq * 64 + to_sq

    def index_to_move(self, index):
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square=from_square, to_square=to_square)


def compute_advantage(rewards, values, gamma=GAMMA, lambda_=LAMBDA):
    """GAE Advantage Computation."""
    rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
    values = np.array(values, dtype=np.float32).reshape(-1, 1)

    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    advantages = np.zeros_like(deltas)

    acc = 0
    for t in reversed(range(len(deltas))):
        acc = deltas[t] + gamma * lambda_ * acc
        advantages[t] = acc

    return advantages.reshape(-1, 1)


def train_actor_critic(actor_critic_model, replay_buffer, batch_size=32):
    """PPO Training Loop for Actor-Critic."""

    states, uci_moves, rewards, next_states, dones = replay_buffer.sample(batch_size)
    if len(states) < batch_size:
        print("Not enough data in replay buffer to train.")
        return None

    actions = [actor_critic_model.uci_to_index(chess.Move.from_uci(move)) for move in uci_moves]
    actions = np.array(actions, dtype=np.int32).reshape(-1, 1)

    # Unpack states
    board_tensors, move_histories, attack_maps, defense_maps, turn_indicators, legal_masks = zip(*states)

    board_tensors = np.array(board_tensors).reshape(batch_size, 8, 8, 20)
    move_histories = np.array(move_histories).reshape(batch_size, 50, 4)
    attack_maps = np.array(attack_maps).reshape(batch_size, 8, 8, 1)
    defense_maps = np.array(defense_maps).reshape(batch_size, 8, 8, 1)
    turn_indicators = np.array(turn_indicators).reshape(batch_size, 1)
    legal_masks = np.array(legal_masks).reshape(batch_size, 64, 64, 1)

    next_board_tensors, next_move_histories, next_attack_maps, next_defense_maps, next_turn_indicators, next_legal_masks = zip(*next_states)

    next_board_tensors = np.array(next_board_tensors).reshape(batch_size, 8, 8, 20)
    next_move_histories = np.array(next_move_histories).reshape(batch_size, 50, 4)
    next_attack_maps = np.array(next_attack_maps).reshape(batch_size, 8, 8, 1)
    next_defense_maps = np.array(next_defense_maps).reshape(batch_size, 8, 8, 1)
    next_turn_indicators = np.array(next_turn_indicators).reshape(batch_size, 1)
    next_legal_masks = np.array(next_legal_masks).reshape(batch_size, 64, 64, 1)

    # Value Estimates
    _, values = actor_critic_model.get_policy_value([
        board_tensors, move_histories, attack_maps, defense_maps, turn_indicators, legal_masks
    ])
    _, next_values = actor_critic_model.get_policy_value([
        next_board_tensors, next_move_histories, next_attack_maps, next_defense_maps, next_turn_indicators, next_legal_masks
    ])

    # Advantage Computation
    advantages = compute_advantage(rewards, values)

    with tf.GradientTape() as tape:
        policy_probs, value_preds = actor_critic_model.get_policy_value([
            board_tensors, move_histories, attack_maps, defense_maps, turn_indicators, legal_masks
        ])

        selected_probs = tf.gather_nd(policy_probs, np.expand_dims(actions, axis=-1), batch_dims=1)

        old_probs = tf.stop_gradient(selected_probs)
        ratio = selected_probs / (old_probs + 1e-8)

        clipped_ratio = tf.clip_by_value(ratio, 1 - EPSILON, 1 + EPSILON)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        value_loss = tf.reduce_mean(tf.square(value_preds - rewards))
        entropy = -tf.reduce_mean(policy_probs * tf.math.log(policy_probs + 1e-8))

        loss = policy_loss + 0.5 * value_loss - ENTROPY_BONUS * entropy

    grads = tape.gradient(loss, actor_critic_model.model.trainable_variables)
    actor_critic_model.model.optimizer.apply_gradients(zip(grads, actor_critic_model.model.trainable_variables))

    return loss.numpy()

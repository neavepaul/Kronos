import tensorflow as tf
from tensorflow import keras
import numpy as np
import chess
from typing import Dict, Tuple
import logging

# Configure logging
logger = logging.getLogger('athena.aegis_net')

# Add MCTS import
from modules.ares.logic.mcts import MCTS

class AegisNet(keras.Model):
    """AlphaZero-style neural network with policy and value heads."""

    def __init__(self, num_filters: int = 256, num_blocks: int = 19):
        logger.info(f"Initializing AegisNet with {num_filters} filters and {num_blocks} residual blocks")
        super().__init__()

        # Initialize MCTS
        self.mcts = MCTS(self, num_simulations=800)

        # Input layers
        self.board_input = keras.layers.Input(shape=(8, 8, 20), name='board_input')
        self.history_input = keras.layers.Input(shape=(50,), name='history_input')
        self.attack_map_input = keras.layers.Input(shape=(8, 8), name='attack_map')
        self.defense_map_input = keras.layers.Input(shape=(8, 8), name='defense_map')

        # Initial processing of inputs
        x = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            padding='same',
            activation='linear',
            name='conv_initial'
        )(self.board_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        # Process attack/defense maps
        attack_map = keras.layers.Reshape((8, 8, 1))(self.attack_map_input)
        defense_map = keras.layers.Reshape((8, 8, 1))(self.defense_map_input)
        maps = keras.layers.Concatenate()([attack_map, defense_map])
        maps = keras.layers.Conv2D(num_filters, 3, padding='same')(maps)
        maps = keras.layers.BatchNormalization()(maps)
        maps = keras.layers.ReLU()(maps)

        # Process move history
        history = keras.layers.Dense(256)(self.history_input)
        history = keras.layers.BatchNormalization()(history)
        history = keras.layers.ReLU()(history)
        history = keras.layers.Reshape((1, 1, 256))(history)
        history = keras.layers.Lambda(lambda x: tf.tile(x, [1, 8, 8, 1]))(history)

        # Merge all features
        merged = keras.layers.Concatenate()([x, maps, history])
        x = keras.layers.Conv2D(num_filters, 1, padding='same')(merged)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        # Residual blocks
        for i in range(num_blocks):
            x = self._residual_block(x, num_filters, f'residual_{i}')

        # === Policy Head Improvement ===
        # Instead of flattening directly from (8x8x32), use Conv2D(2, 1x1) to reduce channels while preserving spatial layout
        # This helps the Dense(4096) layer better distinguish important spatial features
        policy = keras.layers.Conv2D(2, 1, padding='same')(x)
        policy = keras.layers.BatchNormalization()(policy)
        policy = keras.layers.ReLU()(policy)
        policy = keras.layers.Flatten()(policy)  # Now shape is (batch, 8x8x2 = 128)
        policy = keras.layers.Dense(4096, activation='softmax', name='policy')(policy)

        # Value head (unchanged)
        value = keras.layers.Conv2D(32, 3, padding='same')(x)
        value = keras.layers.BatchNormalization()(value)
        value = keras.layers.ReLU()(value)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(256, activation='relu')(value)
        value = keras.layers.Dense(1, activation='tanh', name='value')(value)

        # Create model
        self.alpha_model = keras.Model(
            inputs=[self.board_input, self.history_input,
                    self.attack_map_input, self.defense_map_input],
            outputs=[policy, value]
        )

        # Compile model
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=True
        )
        self.alpha_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mean_squared_error'
            },
            loss_weights={
                'policy': 1.0,
                'value': 1.0
            },
            metrics={
                'policy': [keras.metrics.TopKCategoricalAccuracy(k=1)],
                'value': ['mae']
            }
        )

    def _residual_block(self, x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        skip = x
        x = keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation='linear', name=f'{name}_conv1')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation='linear', name=f'{name}_conv2')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([skip, x])
        x = keras.layers.ReLU()(x)
        return x

    def predict(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        logger.debug(f"Making prediction for position: {board.fen()}")
        from modules.athena.src.utils import fen_to_tensor, encode_move_sequence, get_attack_defense_maps
        fen_tensor = np.expand_dims(fen_to_tensor(board), axis=0)
        move_history = encode_move_sequence([m.uci() for m in board.move_stack])
        move_history = np.expand_dims(move_history, axis=0)
        attack_map, defense_map = get_attack_defense_maps(board)
        attack_map = np.expand_dims(attack_map, axis=0)
        defense_map = np.expand_dims(defense_map, axis=0)

        policy, value = self.alpha_model.predict([
            fen_tensor, move_history, attack_map, defense_map
        ], verbose=0)

        legal_moves = list(board.legal_moves)
        move_probs = {}
        for move in legal_moves:
            move_idx = move.from_square * 64 + move.to_square
            move_probs[move] = policy[0][move_idx]

        prob_sum = sum(move_probs.values())
        if prob_sum > 0:
            move_probs = {m: p / prob_sum for m, p in move_probs.items()}

        logger.debug(f"Prediction complete. Value: {value[0][0]:.3f}, Top move: {max(move_probs.items(), key=lambda x: x[1])[0]}")
        return move_probs, value[0][0]

    def predict_move(self, state: Dict) -> chess.Move:
        logger.debug("Predicting move")
        board_state = np.expand_dims(state['board'], axis=0)
        history = np.expand_dims(state['history'], axis=0)
        attack_map = np.expand_dims(state['attack_map'], axis=0)
        defense_map = np.expand_dims(state['defense_map'], axis=0)
        policy, _ = self.alpha_model.predict([
            board_state, history, attack_map, defense_map
        ], verbose=0)

        board = chess.Board()
        legal_moves = list(board.legal_moves)
        move_probs = []
        for move in legal_moves:
            idx = move.from_square * 64 + move.to_square
            move_probs.append((move, policy[0][idx]))

        if move_probs:
            selected_move = max(move_probs, key=lambda x: x[1])[0]
            logger.debug(f"Selected move: {selected_move}")
            return selected_move
        else:
            logger.warning("No legal moves found in policy distribution")
            return np.random.choice(legal_moves) if legal_moves else None

    def call(self, inputs):
        return self.alpha_model(inputs)

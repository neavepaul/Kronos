import tensorflow as tf
from tensorflow import keras
import numpy as np
import chess
from typing import Dict, Tuple
import logging

logger = logging.getLogger('athena.prometheus_net')

@keras.saving.register_keras_serializable()
class PrometheusNet(keras.Model):
    """Improved AlphaZero-style network with structured policy output."""

    def __init__(self, num_filters: int = 256, num_blocks: int = 19):
        logger.info(f"Initializing PrometheusNet with {num_filters} filters and {num_blocks} residual blocks")
        super().__init__()

        # Input layers
        self.board_input = keras.layers.Input(shape=(8, 8, 20), name='board_input')
        self.history_input = keras.layers.Input(shape=(50,), name='history_input')
        self.attack_map_input = keras.layers.Input(shape=(8, 8), name='attack_map')
        self.defense_map_input = keras.layers.Input(shape=(8, 8), name='defense_map')

        # Initial processing
        x = keras.layers.Conv2D(num_filters, 3, padding='same', activation='linear')(self.board_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        attack_map = keras.layers.Reshape((8, 8, 1))(self.attack_map_input)
        defense_map = keras.layers.Reshape((8, 8, 1))(self.defense_map_input)
        maps = keras.layers.Concatenate()([attack_map, defense_map])
        maps = keras.layers.Conv2D(num_filters, 3, padding='same')(maps)
        maps = keras.layers.BatchNormalization()(maps)
        maps = keras.layers.ReLU()(maps)

        history = keras.layers.Dense(256)(self.history_input)
        history = keras.layers.BatchNormalization()(history)
        history = keras.layers.ReLU()(history)
        history = keras.layers.Reshape((1, 1, 256))(history)
        history = keras.layers.Lambda(lambda h: tf.tile(h, [1, 8, 8, 1]))(history)

        merged = keras.layers.Concatenate()([x, maps, history])
        x = keras.layers.Conv2D(num_filters, 1, padding='same')(merged)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        for i in range(num_blocks):
            x = self._residual_block(x, num_filters, f'residual_{i}')

        # From-square and To-square heads
        flat = keras.layers.GlobalAveragePooling2D()(x)
        from_logits = keras.layers.Dense(64, name='from_logits')(flat)
        to_logits = keras.layers.Dense(64, name='to_logits')(flat)

        # Combine into 64x64 move logits
        policy_logits = keras.layers.Lambda(
                                            lambda t: tf.einsum('bi,bj->bij', t[0], t[1]),
                                            name='policy_einsum'
                                            )([from_logits, to_logits])
        policy_output = keras.layers.Reshape((64, 64), name='policy')(policy_logits)

        # Promotion head (None, Q, R, B, N)
        promotion_output = keras.layers.Dense(5, activation='softmax', name='promotion')(flat)

        # Value head
        value = keras.layers.Conv2D(32, 3, padding='same')(x)
        value = keras.layers.BatchNormalization()(value)
        value = keras.layers.ReLU()(value)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(256, activation='relu')(value)
        value = keras.layers.Dense(1, activation='tanh', name='value')(value)

        self.model = keras.Model(
            inputs=[self.board_input, self.history_input, self.attack_map_input, self.defense_map_input],
            outputs=[policy_output, promotion_output, value]
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'policy': 'categorical_crossentropy',
                'promotion': 'categorical_crossentropy',
                'value': 'mean_squared_error'
            },
            loss_weights={
                'policy': 1.0,
                'promotion': 0.2,
                'value': 1.0
            },
            metrics={
                'policy': ['accuracy'],
                'promotion': ['accuracy'],
                'value': ['mae']
            }
        )

    def _residual_block(self, x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        skip = x
        x = keras.layers.Conv2D(filters, 3, padding='same', activation='linear')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filters, 3, padding='same', activation='linear')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([skip, x])
        x = keras.layers.ReLU()(x)
        return x

    def call(self, inputs):
        return self.model(inputs)

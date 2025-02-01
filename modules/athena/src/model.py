from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense,
    Embedding, Dropout, concatenate, Multiply
)
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

# Transformer Block for Move History
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Function to create and return the model
def get_model():
    # Model Inputs
    fen_input = Input(shape=(8, 8, 20), name="fen_input")
    move_seq_input = Input(shape=(50,), name="move_seq_input")
    legal_moves_input = Input(shape=(64, 64), name="legal_moves_input")

    # CNN for FEN
    x = Conv2D(64, (3,3), activation="relu", padding="same")(fen_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)

    # Transformer for Move History
    y = Embedding(input_dim=5000, output_dim=64, mask_zero=True)(move_seq_input)
    y = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128)(y)
    y = Flatten()(y)
    y = Dense(128, activation="relu")(y)

    # Combine CNN and Transformer branches
    combined = concatenate([x, y])

    # Fully Connected Layers
    z = Dense(256, activation="relu")(combined)
    z = Dense(128, activation="relu")(z)

    # Outputs
    evaluation_output = Dense(1, activation="linear", name="evaluation_output")(z)
    move_from_output = Dense(64, activation="softmax", name="move_from_output")(z)
    move_to_output = Dense(64, activation="softmax", name="move_to_output")(z)

    # Mask illegal moves
    masked_move_from = Multiply()([move_from_output, legal_moves_input])
    masked_move_to = Multiply()([move_to_output, legal_moves_input])

    # Define Model
    model = Model(inputs=[fen_input, move_seq_input, legal_moves_input], outputs=[evaluation_output, masked_move_from, masked_move_to])

    # Compile Model
    model.compile(
        optimizer="adam",
        loss={
            "evaluation_output": "mse",
            "move_from_output": "sparse_categorical_crossentropy",
            "move_to_output": "sparse_categorical_crossentropy"
        }
    )

    return model

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense,
    Embedding, LSTM, concatenate
)

# FEN Input (8x8x20)
fen_input = Input(shape=(8, 8, 20), name="fen_input")

x = Conv2D(64, (3,3), activation="relu", padding="same")(fen_input)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dense(256, activation="relu")(x)

# Move History Input (max_length,)
move_seq_input = Input(shape=(50,), name="move_seq_input")

y = Embedding(input_dim=5000, output_dim=64, mask_zero=True)(move_seq_input)  # 5000 = estimated move vocab size
y = LSTM(128)(y)
y = Dense(128, activation="relu")(y)

# Combine CNN and RNN branches
combined = concatenate([x, y])

# Fully Connected Layers
z = Dense(256, activation="relu")(combined)
z = Dense(128, activation="relu")(z)

# Outputs
evaluation_output = Dense(1, activation="linear", name="evaluation_output")(
    z
)  # Regression (evaluation score)
move_output = Dense(5000, activation="softmax", name="move_output")(
    z
)  # Classification (predict best move)

# Define Model
model = Model(inputs=[fen_input, move_seq_input], outputs=[evaluation_output, move_output])

# Compile Model
model.compile(
    optimizer="adam",
    loss={"evaluation_output": "mse", "move_output": "sparse_categorical_crossentropy"},
    loss_weights=[0.5, 0.5]
)

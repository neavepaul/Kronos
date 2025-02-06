import tensorflow as tf
from tensorflow.keras import Model, layers

# Transformer Block for Move History
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def get_model(vocab_size=700000):
    """
    Returns the TensorFlow model for Athena.
    """
    # **Inputs**
    fen_input = layers.Input(shape=(8, 8, 20), name="fen_input")  # Board State
    move_seq = layers.Input(shape=(50,), dtype=tf.string, name="move_seq")  # Move History
    legal_mask = layers.Input(shape=(64, 64), name="legal_mask")  # Legal Move Mask
    eval_score = layers.Input(shape=(1,), name="eval_score")  # **Stockfish Evaluation Score**

    # **1. Spatial Encoder (CNN + Self-Attention)**
    x = layers.Conv2D(128, 3, activation='gelu', padding='same')(fen_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, activation='gelu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape((64, 256))(x)  # Flatten spatial features
    x = layers.MultiHeadAttention(num_heads=4, key_dim=256)(x, x)

    # **2. Move History Encoder (Transformer)**
    lookup_layer = layers.StringLookup(num_oov_indices=1)  # Maps PFFTTT -> Integer Index
    move_indices = lookup_layer(move_seq)

    emb = layers.Embedding(vocab_size, 128)(move_indices)  # 🔥 **New dynamic move encoding**
    y = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512)(emb)
    y = layers.GlobalAvgPool1D()(y)

    # **3. Evaluation Score Processing**
    eval_scaled = layers.Dense(128, activation="gelu")(eval_score)  # Learnable transformation

    # **4. Fusion**
    fused = layers.Concatenate()([x, y, eval_scaled])
    z = layers.Dense(512, activation='gelu')(fused)
    z = layers.Dropout(0.2)(z)

    # **5. Factorized Move Prediction**
    from_emb = layers.Dense(64, activation='gelu')(z)  # [batch, 64]
    to_emb = layers.Dense(64, activation='gelu')(z)    # [batch, 64]
    logits = layers.Dot(axes=(1, 1))([from_emb, to_emb])  # [batch, 64,64]

    # **6. Legal Move Masking (Lambda Layer)**
    def apply_legal_mask(inputs):
        logits, legal_mask = inputs
        return logits - (1 - legal_mask) * 1e9  # Subtractive masking

    masked_logits = layers.Lambda(apply_legal_mask, name="masked_logits")([logits, legal_mask])
    move_probs = layers.Softmax(name="move_probs")(masked_logits)

    # **7. Criticality Output (For Ares Evaluation, Not Training)**
    criticality = layers.Dense(1, activation='sigmoid', name="criticality")(z)

    # **Define Model**
    model = Model(
        inputs=[fen_input, move_seq, legal_mask, eval_score],
        outputs=[move_probs, criticality]
    )

    # **Compile Model**
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4),
        loss={
            "move_probs": "categorical_crossentropy",  # Predicting a legal move
            "criticality": "binary_crossentropy"  # Determines if Ares should be triggered
        }
    )

    return model

import tensorflow as tf
from tensorflow.keras import Model, layers

MAX_VOCAB_SIZE = 500000


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

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def get_model():
    # Inputs
    fen_input = layers.Input(shape=(8, 8, 20), name="fen_input")  
    move_seq = layers.Input(shape=(50,), dtype=tf.int32, name="move_seq")  
    legal_mask = layers.Input(shape=(64, 64), name="legal_mask")  
    turn_indicator = layers.Input(shape=(1,), dtype=tf.float32, name="turn_indicator")  
    eval_score = layers.Input(shape=(1,), dtype=tf.float32, name="eval_score")  

    # Spatial Encoder (CNN)
    x = layers.Conv2D(128, 3, activation='gelu', padding='same')(fen_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, activation='gelu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape((64, 256))(x)  
    x_pooled = layers.GlobalAvgPool1D()(x)  

    # Move History Encoder (Transformer)
    emb = layers.Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=128)(move_seq)  
    y = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512)(emb, training=True)
    y = layers.GlobalAvgPool1D()(y)

    # Fusion
    eval_scaled = layers.Dense(128, activation="gelu")(eval_score)  
    turn_scaled = layers.Dense(32, activation="gelu")(turn_indicator)
    fused = layers.Concatenate()([x_pooled, y, eval_scaled])  

    z = layers.Dense(512, activation='gelu')(fused)
    z = layers.Dropout(0.2)(z)

    # Move Output (PFFTTU format)
    move_output = layers.Dense(6, activation='linear', name="move_output")(z)

    # Criticality Score (Should Ares Search?)
    criticality = layers.Dense(1, activation='sigmoid', name="criticality")(z)

    # Define Model
    model = Model(inputs=[fen_input, move_seq, legal_mask, eval_score], outputs=[move_output, criticality])

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4),
        loss={"move_output": "mse", "criticality": "binary_crossentropy"}
    )

    return model

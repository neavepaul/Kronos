import tensorflow as tf
from tensorflow.keras import layers, Model

class TransformerBlock(layers.Layer):
    """Self-Attention Block for Chess Board Encoding."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.norm2(out1 + ffn_output)

class TransformerBoardEncoder(Model):
    """Encodes Chess Board States Using Transformers."""
    
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=512, num_layers=4):
        super(TransformerBoardEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.embedding = layers.Dense(embed_dim)  # Converts board input to embedding
        self.pos_encoding = self.positional_encoding(8 * 8, embed_dim)  # Fix shape mismatch

        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ]

        self.flatten = layers.GlobalAveragePooling1D()  # Extract board features

    def call(self, board_tensor):
        # Convert board to embedding
        x = self.embedding(board_tensor)  # Shape: (batch, 8, 8, embed_dim)

        # Flatten spatial dimensions to (batch, 64, embed_dim)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, 64, self.embed_dim))  

        # Ensure positional encoding is broadcastable
        x += tf.reshape(self.pos_encoding, (1, 64, self.embed_dim))  

        # Apply transformer layers
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        return self.flatten(x)

    def positional_encoding(self, position, d_model):
        """Generates positional encodings for Transformer input."""

        # Ensure `d_model` is float32 for calculations
        d_model_float = tf.cast(d_model, tf.float32)  

        # Compute angle rates
        angle_rates = 1 / tf.pow(10000.0, (2.0 * (tf.cast(tf.range(d_model), tf.float32) // 2)) / d_model_float)

        # Compute angles
        angles = tf.cast(tf.range(position), tf.float32)[:, tf.newaxis] * angle_rates[tf.newaxis, :]

        # Apply sin and cos functions
        pos_encoding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)

        # Ensure correct indexing type (cast `d_model` to int)
        return tf.cast(pos_encoding[:, :tf.cast(d_model, tf.int32)], dtype=tf.float32)



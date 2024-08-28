from tensorflow.keras.layers import Conv1D, Attention
from tensorflow.keras import layers

class AAConvTransformerAgent:
    def aa_conv_block(self, inputs, filters, kernel_size, num_heads, head_size, dropout=0.1):
        conv_out = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(inputs)
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(conv_out, conv_out)
        attn_out = layers.LayerNormalization(epsilon=1e-6)(attn_out)
        return conv_out + attn_out

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        x = self.aa_conv_block(inputs, filters=64, kernel_size=3, num_heads=num_heads, head_size=head_size, dropout=dropout)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res
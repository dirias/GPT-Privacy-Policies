import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = Dense(self.head_dim * heads, activation=None)
        self.keys = Dense(self.head_dim * heads, activation=None)
        self.queries = Dense(self.head_dim * heads, activation=None)
        self.fc_out = Dense(embed_size)

    def call(self, x):
        values, keys, queries, mask = x

        N = tf.shape(values)[0]  
        value_len, key_len, query_len = tf.shape(values)[1], tf.shape(keys)[1], tf.shape(queries)[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        values = tf.reshape(values, (N, value_len, self.heads, self.head_dim))
        
        keys = tf.reshape(keys, (N, key_len, self.heads, self.head_dim))
        queries = tf.reshape(queries, (N, query_len, self.heads, self.head_dim))

        # Scaled dot-product attention
        attention = tf.einsum("nqhd,nkhd->nhqk", queries, keys)
        """
        n: Dimensión del lote.
        q: Dimensión que representa las consultas.
        h: Dimensión que representa las cabezas.
        d: Dimensión que representa las dimensiones de la cabeza (tamaño del subvector).
        """
        if mask is not None:
            # Convert mask to float type
            mask = tf.cast(mask, dtype=tf.float32)
            
            # Expand the mask dimensions to match the attention scores
            mask_expanded = tf.expand_dims(mask, axis=1)  # shape: [batch_size, 1, seq_len]
            mask_expanded = tf.expand_dims(mask_expanded, axis=1)  # shape: [batch_size, 1, 1, seq_len]
            attention += (mask_expanded * -1e9)  # broadcast mask to have shape [batch_size, heads, query_len, key_len]


        attention = tf.nn.softmax(attention / tf.math.sqrt(tf.cast(self.head_dim, tf.float32)), axis=3)
        out = tf.einsum("nhql,nlhd->nqhd", attention, values)
        out = tf.reshape(out, (N, query_len, self.heads * self.head_dim))

        # Concatenate heads and pass through final feed-forward layer
        return self.fc_out(out)
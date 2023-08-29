import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

from MultiHeadSelfAttention import MultiHeadSelfAttention

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6) #0.000001
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6) # 0.000001

        self.feed_forward = tf.keras.Sequential([
            Dense(forward_expansion * embed_size, activation="relu"),
            Dense(embed_size),
        ])
        self.dropout = Dropout(dropout)

    def call(self, x):
        value, key, query, mask = x
        attention = self.attention([value, key, query, mask])  # Pass mask as the last argument
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        return self.norm2(forward + x)
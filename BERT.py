import tensorflow as tf
from tensorflow.keras.layers import Embedding

# Custom Classes

from TransformerBlock import TransformerBlock

class BERT(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, embed_size)
        self.position_embedding = Embedding(max_length, embed_size)
        # Renaming self.layers to self.transformer_blocks
        self.transformer_blocks = [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]

    def call(self, x):
        input_tensor, mask = x
        N, seq_length = input_tensor.shape
        positions = tf.range(0, seq_length)
        out = self.embedding(input_tensor) 
        positions = self.position_embedding(positions)
        out += positions
        for block in self.transformer_blocks:
            out = block([out, out, out, mask])
        return out
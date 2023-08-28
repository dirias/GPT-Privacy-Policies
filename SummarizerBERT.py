
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Custom Classes
from BERT import BERT

class SummarizerBERT(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length):
        super(SummarizerBERT, self).__init__()
        self.bert = BERT(vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length)
        self.score_layer = Dense(1, activation=None)

    def call(self, x):
        input_tensor, mask = x
        embeddings = self.bert([input_tensor, mask])
        
        # Extract [CLS] embeddings (assuming [CLS] is at position 0 for each sentence)
        cls_embeddings = embeddings[:, 0, :]
        
        # Score each sentence
        scores = self.score_layer(cls_embeddings)
        
        return scores
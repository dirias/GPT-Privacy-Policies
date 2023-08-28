#!pip install transformers

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Dropout
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import requests



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
        print("Values shape:", values.shape)
        print("Keys shape:", keys.shape)
        print("Queries shape:", queries.shape)

        # Scaled dot-product attention
        # Scaled dot-product attention
        attention = tf.einsum("nqhd,nkhd->nhqk", queries, keys)
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

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.feed_forward = tf.keras.Sequential([
            Dense(forward_expansion * embed_size, activation="relu"),
            Dense(embed_size),
        ])
        self.dropout = Dropout(dropout)

    def call(self, x):
        value, key, query, mask = x
        attention = self.attention([value, key, query, mask])  # Pass mask as the last argument
        print("Attention shape:", attention.shape)
        print("Query shape:", query.shape)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        return self.norm2(forward + x)

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
            print("BERT output shape after block:", out.shape)
        return out
    
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

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
VOCAB_SIZE = 30522  # This is for BERT base
EMBED_SIZE = 768
MAX_LENGTH = 512
NUM_LAYERS = 12
HEADS = 12
FORWARD_EXPANSION = 4
DROPOUT = 0.1

# 1. Data Preparation


url = "https://raw.githubusercontent.com/citp/privacy-policy-historical/master/0/00/000/000domains.com.md"
response = requests.get(url)
text = response.text

# Splitting by two newlines to consider a paragraph
texts = [para.strip() for para in text.split('\n\n') if para]

# Define key terms/phrases
key_terms = ["data collection", "third-party", "cookies", "personal information", "disclosure", "security", "rights", "retention"]

# Now, let's label our original paragraphs based on the presence of key terms.
labels = [1 if any(term in paragraph.lower() for term in key_terms) else 0 for paragraph in texts]

# Split data into training and validation
texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 2. Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(texts_train, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="tf")
val_encodings = tokenizer(texts_val, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="tf")
import pdb; pdb.set_trace()
# 3. Model Compilation
model = SummarizerBERT(VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS, HEADS, FORWARD_EXPANSION, DROPOUT, MAX_LENGTH)
optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 4. Training
print('Fitting ------------------------------------->')
model.fit([train_encodings['input_ids'], train_encodings['attention_mask']], np.array(labels_train),
        validation_data=([val_encodings['input_ids'], val_encodings['attention_mask']], np.array(labels_val)),
        batch_size=BATCH_SIZE, epochs=EPOCHS)
# 5. Prediction
print('Prediction ------------------------------------->')

def generate_summary(policy_text, model, tokenizer, summary_length=3):
    # Split the policy text into paragraphs
    paragraphs = [para.strip() for para in policy_text.split('\n\n') if para]
    
    # Tokenize the paragraphs
    encodings = tokenizer(paragraphs, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="tf")
    
    # Predict importance scores for each paragraph
    import pdb; pdb.set_trace()
    predictions = model.predict([encodings['input_ids'], encodings['attention_mask']])
    predictions = tf.squeeze(predictions).numpy()
    
    # Get indices of top paragraphs
    top_paragraph_indices = predictions.argsort()[-summary_length:][::-1]
    
    # Extract these paragraphs from the policy text
    summarized_paragraphs = [paragraphs[i] for i in top_paragraph_indices]
    
    # Combine paragraphs to form the summary
    summary = '\n\n'.join(summarized_paragraphs)
    
    return summary

# Test using a privacy policy excerpt
policy_text = """
At Company XYZ, we are deeply committed to safeguarding the privacy and personal information of our users. 
We recognize the trust you place in us when you share your data, and we consider it our topmost responsibility to ensure its security.
This Privacy Policy has been designed to inform our users about the practices we employ concerning data collection, handling, and storage. 
The primary aim of collecting data is to continuously enhance our services, making them tailored to our user's preferences. We believe that by understanding our users better, we can offer more intuitive features and ensure a seamless user experience.
It's essential for our users to note that the data we collect is stored with top-tier encryption protocols. We have partnered with leading cybersecurity firms to ensure that the infrastructure hosting user data remains impenetrable to unauthorized access. Moreover, Company XYZ firmly stands by the principle that user data is sacred and, under no circumstances, is shared, sold, or leased to third-party companies, advertisers, or data brokers.
By choosing to use our services, you inherently consent to the practices described in this policy. 
It is always advisable to review this policy periodically, as we may update it to reflect changes in legal or operational requirements. 
Lastly, we are always available to address any questions, concerns, or suggestions you may have. 
Feel free to reach out to our dedicated support team, who will be more than happy to assist you.
Thank you for placing your trust in Company XYZ. Together, let's create a digital experience that respects and celebrates privacy.
"""

summary = generate_summary(policy_text, model, tokenizer, summary_length=1)
print(summary)

#!pip install transformers

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

# Custom Classes
from utils import generate_summary, generate_input_text
from SummarizerBERT import SummarizerBERT

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

# Define key terms/phrases
KEY_TERMS = ["data collection", "third-party", "cookies", "personal information", "disclosure", "security", "rights", "retention"]

# 1. Data Preparation
# Split data into training and validation
texts_train, texts_val, labels_train, labels_val = generate_input_text(KEY_TERMS)

# 2. Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(texts_train, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="tf")
val_encodings = tokenizer(texts_val, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="tf")

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

# 5. Exporting the model
model.save('trained_model')

# 5. Prediction
print('Prediction ------------------------------------->')

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
summary = generate_summary(policy_text, model, tokenizer, MAX_LENGTH,summary_length=1)
print(summary)


import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_SOURCE_URL = "https://raw.githubusercontent.com/citp/privacy-policy-historical/master/0/00/000/000domains.com.md"

def get_training_data(url = DATA_SOURCE_URL):
    response = requests.get(url)
    text = response.text

    # Splitting by two newlines to consider a paragraph
    texts = [para.strip() for para in text.split('\n\n') if para]

    return texts

def generate_input_text(KEY_TERMS):

    # Splitting by two newlines to consider a paragraph
    texts = get_training_data(DATA_SOURCE_URL)

    # Now, let's label our original paragraphs based on the presence of key terms.
    labels = [1 if any(term in paragraph.lower() for term in KEY_TERMS) else 0 for paragraph in texts]

    # Split data into training and validation
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    return texts_train, texts_val, labels_train, labels_val
    
def generate_summary(policy_text, model, tokenizer, MAX_LENGTH,summary_length=3):
    # Split the policy text into paragraphs, line by line
    paragraphs = [line.strip() for line in policy_text.split('\n') if line]
    # Tokenize the paragraphs
    encodings = tokenizer(paragraphs, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="tf")
    # To see decoded information
    # tokenizer.decode(encodings['input_ids'][0].numpy(), skip_special_tokens=True)

    # Predict importance scores for each paragraph
    predictions = model.predict([encodings['input_ids'], encodings['attention_mask']])
    import pdb; pdb.set_trace
    predictions = tf.squeeze(predictions).numpy()
    
    # Get indices of top paragraphs
    top_paragraph_indices = predictions.argsort()[-summary_length:][::-1]
    
    # Extract these paragraphs from the policy text
    summarized_paragraphs = [paragraphs[i] for i in top_paragraph_indices]
    
    # Combine paragraphs to form the summary
    summary = '\n\n'.join(summarized_paragraphs)
    
    return summary

import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics import f1_score
import re

nltk.download('punkt')

DATA_SOURCE_URL = "https://raw.githubusercontent.com/citp/privacy-policy-historical/master/"
GITHUB_API_URL = "https://api.github.com/repos/citp/privacy-policy-historical/git/trees/master?recursive=1"

def get_file_content(file_url):
    response = requests.get(file_url)
    return response.text

def get_md_files_from_repo(api_url=GITHUB_API_URL):
    response = requests.get(api_url)
    data = response.json()

    if 'tree' not in data:
        print("Failed to fetch the repository structure.")
        return []

    md_files = [item for item in data['tree'] if item['path'].endswith('.md')]

    return md_files

def get_training_data(max_files=None):
    md_files = get_md_files_from_repo()
    
    # Limit the number of files based on the max_files parameter
    md_files = md_files[:max_files] if max_files else md_files
    
    policies = []
    for file_info in md_files:
        file_content = get_file_content(DATA_SOURCE_URL + file_info['path'])
        policies.append(file_content)

    return policies

def save_data(data: list, text_file='data.txt'):
    with open(text_file, 'w') as f:
        for line in data:
            f.write(line + '\n')

def strip_markdown(md_content: str):
    # Remove headers (#, ##, ###, etc.)
    md_content = re.sub(r'#+\s', '', md_content)
    
    # Remove list numbers and bullet points (1., 2., *, etc.)
    md_content = re.sub(r'^\d+\.\s|\*\s', '', md_content, flags=re.MULTILINE)
    
    # Remove bold, italic, etc. (**text**, *text*, __text__, _text_)
    md_content = re.sub(r'(\*\*|__|\*|_)(.*?)(\*\*|__|\*|_)', r'\2', md_content)
    
    # Remove links [text](url)
    md_content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', md_content)
    
    # Remove inline code `code`
    md_content = re.sub(r'`(.*?)`', r'\1', md_content)
    
    # Remove blockquotes
    md_content = re.sub(r'^>\s', '', md_content, flags=re.MULTILINE)

    return md_content

def generate_input_text(KEY_TERMS):

    # Splitting by two newlines to consider a paragraph
    data = get_training_data(max_files=10)
    formatted_data = strip_markdown(''.join(data))
    texts = sent_tokenize(formatted_data)
    save_data(texts)
    # Now, let's label our original paragraphs based on the presence of key terms.
    labels = [1 if any(term in paragraph.lower() for term in KEY_TERMS) else 0 for paragraph in texts]

    # Split data into training and validation
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    return texts_train, texts_val, labels_train, labels_val
    
def generate_summary(policy_text, model, tokenizer, MAX_LENGTH,summary_length=3):
    # Split the policy text into paragraphs, line by line
    paragraphs = sent_tokenize(policy_text)
    # Tokenize the paragraphs
    encodings = tokenizer(paragraphs, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="tf")
    # To see decoded information
    # tokenizer.decode(encodings['input_ids'][0].numpy(), skip_special_tokens=True)

    # Predict importance scores for each paragraph
    predictions = model.predict([encodings['input_ids'], encodings['attention_mask']])
    predictions = tf.squeeze(predictions).numpy()
    
    # Get indices of top paragraphs
    top_paragraph_indices = predictions.argsort()[-summary_length:][::-1]
    
    # Extract these paragraphs from the policy text
    summarized_paragraphs = [paragraphs[i] for i in top_paragraph_indices]
    
    # Combine paragraphs to form the summary
    summary = '\n\n'.join(summarized_paragraphs)
    
    return summary

# Define a custom F1-Score metric function
def f1_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    return f1_score(y_true, y_pred, average='weighted')
import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer

from utils import generate_summary

# Load the model and tokenizer
MAX_LENGTH = 512

loaded_model = tf.keras.models.load_model('trained_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# App title and description
st.set_page_config(page_title="GPT - Privacy Policies",
                   page_icon="📜",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.title("GPT Privacy Policies Summarizer 📜")
st.write(
    "Welcome to the Privacy Policies Summarizer! This tool uses advanced NLP techniques to give you the gist of a long text.")
st.write("---")

# Sidebars for additional info and developer details
st.sidebar.header("Developer Info")
st.sidebar.write("Developer: Didier Irias Méndez")
st.sidebar.write("- [LinkedIn](https://www.linkedin.com/in/didier-irias-m%C3%A9ndez-4ba593147/)")
st.sidebar.write("- [Github](https://github.com/dirias)")


# Main content
st.subheader("Input")
user_input = st.text_area("Enter the text you want to summarize:", height=300)

if st.button("Summarize"):
    if user_input:
        try:
            with st.spinner("Generating summary..."):
                summary = generate_summary(
                    user_input, loaded_model, tokenizer, MAX_LENGTH, summary_length=4)
            st.subheader("Summary")
            st.success(summary)
        except Exception as ex:
            st.error(f'There was an error while trying to make the summary: {ex}')

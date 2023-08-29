import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer

from utils import generate_summary

MAX_LENGTH = 512

loaded_model = tf.keras.models.load_model('trained_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


st.title("GPT - Privacy Policies ")

st.write("Developer: Didier Irias Méndez")
st.write("Curso de aprendizaje automático")

# Create a text input widget
user_input = st.text_input("Enter the text you want to summary:")

# Display the entered text (if any)
if user_input:
    try:
        summary = generate_summary(user_input, loaded_model, tokenizer, MAX_LENGTH, summary_length=4)
        st.write(f"The summarized text is \n: {summary}")
    except Exception as ex:
        raise (f'There was an erro while trying to make the summary: {ex}')
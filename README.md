# GPT-Privacy-Policies

This repo creates a small GPT system along with a web interface to generate summaries about Privacy Policies using a GPT architecture, simulating a BERT AI model.

## Files.
- Main.ipynb / Main.py: This is the main file, which uses specials KEY_TERMS to identify which part of the paragraph be aware of.
- SummarizerBERT.py: This class inherits from the BERT one and it is intended to get the embeddings and provide a score based on the key terms.
- BERT.py: This class simulates a small Bert architecture, it creates the embeddings, positional embeddings, and the transformer block
- TransformerBlock.py: This creates a Transformer architecture, it implements the MultiHeadSelfAttention system to focus on the desired data based on the KEY_TERMS.
- MultiHeadSelfAttention.py: This implements the MultiHeadSelfAttention system using the embedding values, keys, and queries.
- utils.py:
- app.py: This will run a small Python app that activates a browser interface to interact with the model.

## How to run the AI model
- Since the model is too heavy to be stored in GitHub, you can run the Main.ipynb / Main.pyfiles to generate a new one, it will take about 40 ~ 50 minutes long 😄...
- After that, run the app.py with <code>streamlit run app.py  </code>, which will open a web interface to interact with the AI model.

## Data Source
- The data source to train the model comes from https://github.com/citp/privacy-policy-historical/tree/master/0.
- The utils.py has a function called <code>generate_input_text</code> which takes a <code>max_files</code> parameter to get the <code>n</code> first .md files found in the datasource repo.
- Then, the <code>strip_markdown</code> functions remove the .md file formatting to avoid noise while training the model.

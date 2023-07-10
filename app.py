##Libraries
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import openai
import time
from openai.embeddings_utils import distances_from_embeddings
import tiktoken
# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")
openai.api_key = st.secrets["openai_key"]

df = pd.read_parquet('data_with_embeddings.parquet')
embedding_model = 'text-embedding-ada-002'
llm_model = "text-davinci-003"

#Functions
def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine=embedding_model)['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
        question,
        df,
        model=llm_model,
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=150,
        stop_sequence=None
    ):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know, please call on 1800-001-002\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

##App framework
st.image('exl_logo.png', width = 150)
st.title('Customer Chatbot')

prompt = st.text_area('Please enter your query', height = 5)


if st.button('Ask'):
    progress_text = "Getting back with the information"
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(2)
        my_bar.progress(percent_complete + 1, text=progress_text)
        model_output =  answer_question(prompt, df, debug = False)
        if type(model_output) == type('Check'):
            st.write(model_output)
            my_bar.progress(100)
            break
    

"""
File name: oracle_bot.py
Author: Luigi Saetta
Date created: 2023-12-17
Date last modified: 2024-02-27
Python Version: 3.9

Description:
    This module provides the chatbot UI for the RAG demo 

Usage:
    run with: streamlit run oracle_bot.py

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to buil a RAG solution,
    where all the data (text + embeddings) are stored in Oracle DB 23c 

Warnings:
    This module is in development, may change in future versions.
"""

import logging
import time
import streamlit as st

# to use the create_query_engine
from factory import get_lang_chain

#
# Configs
#


def reset_conversation():
    st.session_state.messages = []


# defined here to avoid import of streamlit in other module
# cause we need here to use @cache
@st.cache_resource
def create_query_engine(verbose=False):
    query_engine = get_lang_chain(verbose=verbose)

    # token_counter keeps track of the num. of tokens
    return query_engine


# to format output with references
def format_output(response):
    output = response

    # TODO format to add references
    return output


#
# Main
#

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

st.title("OCI Bot powered by Generative AI")

# Added reset button
st.button("Clear Chat History", on_click=reset_conversation)

# Initialize chat history

if "messages" not in st.session_state:
    reset_conversation()

# init RAG
with st.spinner("Initializing RAG chain..."):
    # I have added the token counter to count token
    # I've done this way because I marked the function with @cache
    # but there was a problem with the counter. It works if it is created in the other module
    # and returned here where I print the results for each query

    # here we create the query engine
    query_engine = create_query_engine(verbose=False)


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if question := st.chat_input("Hello, how can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    # here we call OCI genai...

    try:
        logging.info("Calling RAG chain..")

        with st.spinner("Waiting..."):
            tStart = time.time()

            # Here we call the entire chain !!!
            response = query_engine.invoke(question)

            tEla = time.time() - tStart
            logging.info(f"Elapsed time: {round(tEla, 1)} sec.")

        # Display assistant response in chat message container
        with st.chat_message("assistant"):

            output = format_output(response)

            st.markdown(output)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": output})

    except Exception as e:
        logging.error("An error occurred: " + str(e))
        st.error("An error occurred: " + str(e))

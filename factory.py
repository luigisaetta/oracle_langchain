"""
File name: factory.py
Author: Luigi Saetta
Date created: 2024-02-27
Date last modified: 2024-02-27
Python Version: 3.9

Description:
    This module provides the class to integrate Oracle
    DB Vector Store in LangChain.
    This version uses only one table (chunks + vecs)

Inspired by:
    

Usage:
    Import this module into other scripts to use its functions. 
    Example:
        from factory import get_lang_chain
        rag_chain = get_lang_chain()

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to build a RAG solution,
    where all the data (text + embeddings) are stored in Oracle DB 23c
    Modified (25/02) to pass the Embed model and not the fuction

Warnings:
    This module is in development, may change in future versions.

"""

from oracle_vector_db_lc import OracleVectorStore

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.llms import OCIGenAI

from utils import format_docs, print_configuration
from config import EMBED_MODEL, GENAI_MODEL, ENDPOINT, TOP_K, TEMPERATURE, MAX_TOKENS

from config_private import COMPARTMENT_OCID


def get_embed_model():
    embed_model = OCIGenAIEmbeddings(
        auth_type="API_KEY",
        model_id=EMBED_MODEL,
        service_endpoint=ENDPOINT,
        compartment_id=COMPARTMENT_OCID,
    )
    return embed_model


def get_llm():
    llm = OCIGenAI(
        auth_type="API_KEY",
        model_id=GENAI_MODEL,
        service_endpoint=ENDPOINT,
        compartment_id=COMPARTMENT_OCID,
        model_kwargs={
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
    )
    return llm


def get_lang_chain(verbose=False):

    print_configuration()

    embed_model = get_embed_model()

    llm = get_llm()

    # the prompt. This is OK for Cohere
    prompt = hub.pull("rlm/rag-prompt")

    v_store = OracleVectorStore(
        embedding=embed_model, collection_name="ORACLE_KNOWLEDGE", verbose=verbose
    )

    retriever = v_store.as_retriever(search_kwargs={"k": TOP_K})

    # using LangChain LCEL language
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

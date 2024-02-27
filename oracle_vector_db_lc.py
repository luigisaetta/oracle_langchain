"""
File name: oracle_vector_db_lc.py
Author: Luigi Saetta
Date created: 2024-01-17
Date last modified: 2024-02-26
Python Version: 3.9

Description:
    This module provides the class to integrate Oracle
    DB Vector Store in LangChain.
    This version uses only one table (chunks + vecs)

Inspired by:
    

Usage:
    Import this module into other scripts to use its functions. 
    Example:
        from oracle_vector_db_lc import OracleVectorStore
        
        v_store = OracleVectorStore(embedding=embed_model,
                            verbose=True)

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

from __future__ import annotations
from tqdm.auto import tqdm

import time
import array
import numpy as np

import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

import oracledb
import logging

# load configs from here
from config_private import DB_USER, DB_PWD, DB_HOST_IP, DB_SERVICE

# 64 or 32
from config import EMBEDDINGS_BITS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

VST = TypeVar("VST", bound="VectorStore")


#
# supporting functions
#
def make_dsn():
    DSN = DB_HOST_IP + "/" + DB_SERVICE

    return DSN


def get_type_from_bits():
    if EMBEDDINGS_BITS == 64:
        type = "d"
    else:
        type = "f"

    return type


def oracle_query(
    embed_query: List[float], collection_name: str, top_k: int = 3, verbose=False
) -> List[Document]:
    """
    Executes a query against an Oracle database to find the top_k closest vectors to the given embedding.

    History:
        23/12/2023: modified to return some metadata (ref)
    Args:
        embed_query (List[float]): A list of floats representing the query vector embedding.
        top_k (int, optional): The number of closest vectors to retrieve. Defaults to 2.
        verbose (bool, optional): If set to True, additional information about the query and execution time will be printed. Defaults to False.

    Returns:
        List[Document]
    """
    tStart = time.time()

    # build the DSN from data taken from config.py
    DSN = make_dsn()

    try:
        with oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN) as connection:
            with connection.cursor() as cursor:
                # 'f' single precision 'd' double precision
                btype = get_type_from_bits()

                array_query = array.array(btype, embed_query)

                # changed select adding books (29/12/2023)
                select = f"""select C.id, C.CHUNK, C.REF, 
                            ROUND(VECTOR_DISTANCE(C.VEC, :1, DOT), 3) as d 
                            from {collection_name} C
                            order by d
                            FETCH FIRST {top_k} ROWS ONLY"""

                if verbose:
                    logging.info(f"select: {select}")

                cursor.execute(select, [array_query])

                rows = cursor.fetchall()

                result_docs = []
                node_ids = []
                similarities = []

                # prepare output
                for row in rows:
                    clob_pointer = row[1]
                    full_clob_data = clob_pointer.read()

                    # 29/12: added book_name to metadata
                    result_docs.append(
                        # pack in the expected format
                        Document(
                            page_content=full_clob_data,
                            metadata={"source": row[2]},
                        )
                    )
                    # not used, for now
                    node_ids.append(row[0])
                    similarities.append(row[3])

    except Exception as e:
        logging.error(f"Error occurred in oracle_query: {e}")

        return None

    tEla = time.time() - tStart

    if verbose:
        logging.info(f"Query duration: {round(tEla, 1)} sec.")

    return result_docs


#
# OracleVectorStore
#
class OracleVectorStore(VectorStore):
    # To avoid problems with OCI GenAI Embeddings
    # where Cohere has a limit on 96
    _BATCH_SIZE = 90

    _DEFAULT_COLLECTION_NAME = "CHUNKS_VECTORS"

    def __init__(
        self,
        embedding: Embeddings,
        *,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[Dict] = None,
        client: Optional[Any] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.verbose = verbose
        # the name fro the Oracle DB table
        self.collection_name = collection_name

        self._embedding_model = embedding

        self.override_relevance_score_fn = relevance_score_fn

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embedding: Text embedding model to use.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        raise NotImplementedError("add_texts method must be implemented...")

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding_model

    #
    # similarity_search
    #
    def similarity_search(
        self, query: str, k: int = 3, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""

        if self.verbose:
            logging.info(f"top_k: {k}")
            logging.info("")

        # 1. embed the query
        embed_query = self._embedding_model.embed_query(query)

        # 2. invoke oracle_query, return List[Document]
        result_docs = oracle_query(
            embed_query=embed_query,
            collection_name=self.collection_name,
            top_k=k,
            verbose=self.verbose,
        )

        return result_docs

    #
    # This function enable to load a table from scratch, with
    # texts and embeddings... then you can query
    #
    @classmethod
    def from_documents(
        cls: Type[OracleVectorStore],
        documents: List[Document],
        embedding: Embeddings,
        # the name for Oracle DB table
        collection_name: str,
        verbose=False,
        **kwargs: Any,
    ) -> OracleVectorStore:
        """Return VectorStore initialized from documents and embeddings.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided Oracle DB with AI
               Vetor Search

        This is intended to be a quick way to get started.

        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        refs = [metadata["source"] for metadata in metadatas]

        # compute embeddings
        # here we use correctly embed_documents
        # (26/02) I'll handle directly inside here the batching
        logging.info("Compute embeddings...")

        batch_size = cls._BATCH_SIZE

        if len(texts) > batch_size:
            embeddings = []

            # do in batch
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i : i + batch_size]

                # here we compute embeddings for a batch
                embeddings_batch = embedding.embed_documents(batch)

                # add to the final list
                embeddings.extend(embeddings_batch)
        else:
            # single batch
            embeddings = embedding.embed_documents(texts)

        # trasform output in numpy array
        embeddings = np.array(embeddings)

        # save in db
        tot_errors = 0

        DSN = make_dsn()

        with oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN) as connection:
            with connection.cursor() as cursor:
                logging.info("Saving texts, embeddings to DB...")

                id = 0
                for vector, chunk, ref in zip(tqdm(embeddings), texts, refs):
                    id += 1

                    # 'f' single precision 'd' double precision
                    btype = get_type_from_bits()

                    input_array = array.array(btype, vector)

                    try:
                        # insert single row
                        cursor.execute(
                            f"insert into {collection_name} (ID, CHUNK, VEC, REF) values (:1, :2, :3, :4)",
                            [id, chunk, input_array, ref],
                        )
                    except Exception as e:
                        logging.error("Error in save embeddings...")
                        logging.error(e)
                        tot_errors += 1

            connection.commit()

            logging.info(f"Tot. errors in save_embeddings: {tot_errors}")

            # beware: here we're passing the model... this cls can be
            # after used for query
            return cls(
                embedding=embedding, collection_name=collection_name, verbose=verbose
            )

    @classmethod
    def from_texts(
        cls: Type[OracleVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> OracleVectorStore:
        """Return VectorStore initialized from texts and embeddings."""
        raise NotImplementedError("from_texts method must be implemented...")

    # this is an utility function to clean the table during tests
    # be aware it deletes all the record in the mentioned table
    @classmethod
    def drop_collection(cls, collection_name: str):

        DSN = make_dsn()

        with oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN) as connection:
            with connection.cursor() as cursor:
                sql = f"""
                    BEGIN
                        execute immediate 'drop table {collection_name}';
                    END;
                    """

                cursor.execute(sql)

                logging.info(f"{collection_name} dropped!!!")

    @classmethod
    def create_collection(cls, collection_name: str):
        DSN = make_dsn()

        with oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN) as connection:
            with connection.cursor() as cursor:
                create_sql = f"""create table {collection_name}
                    (ID NUMBER NOT NULL,
                    CHUNK CLOB,
                    VEC  VECTOR(1024, FLOAT64),
                    REF VARCHAR2(1000),
                    PRIMARY KEY ("ID")
                    )
                """
                sql = f"""
                    BEGIN
                        execute immediate '{create_sql}';
                    END;
                    """

                cursor.execute(sql)

                logging.info(f"{collection_name} created!!!")

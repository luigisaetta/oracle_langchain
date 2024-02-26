#
# To load the Vector Store
#
import logging
from glob import glob
import pandas as pd

# to load and split txt documents
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# to compute embeddings vectors
from langchain_community.embeddings import OCIGenAIEmbeddings

# the class to integrate OCI AI Vector Search with LangChain
from oracle_vector_db_lc import OracleVectorStore

from config import EMBED_MODEL, ENDPOINT, CHUNK_SIZE, TXT_DIR, REF_FILE

from config_private import COMPARTMENT_OCID

# local configs
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

#
# Helper functions
#


# find the url from the file name, using references.csv
def find_ref(df, f_name):
    condition = df["file_name"] == f_name

    ref = df.loc[condition]["url"].values[0]

    return ref


def set_url_in_docs(docs, df_ref):
    docs = docs.copy()
    for doc in docs:
        # remove txt from file_name
        file_name = doc.metadata["source"]
        only_name = file_name.split("/")[-1]
        # find the url from the csv
        ref = find_ref(df_ref, only_name)

        doc.metadata["source"] = ref

    return docs


#
# Main
#
logging.info("")
logging.info("Start loading...")

# this is the file list containing the Knowledge base
file_list = sorted(glob(TXT_DIR + "/" + "*.txt"))

logging.info(f"There are {len(file_list)} files to be loaded...")

# read all references (url)
df_ref = pd.read_csv(REF_FILE)

# load txt and splits in chunks
# with TextLoader it is fast
# documents read not yet splitted
origin_docs = DirectoryLoader(
    TXT_DIR, glob="**/*.txt", show_progress=True, loader_cls=TextLoader
).load()

# replace the f_name with the reference (url)
origin_docs = set_url_in_docs(origin_docs, df_ref)

# split docs in chunks
text_splitter = RecursiveCharacterTextSplitter(
    # thse params must be adapted to Knowledge base
    chunk_size=CHUNK_SIZE,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

docs_splitted = text_splitter.split_documents(origin_docs)

logging.info(f"We have splitted docs in {len(docs_splitted)} chunks...")

# prepare a clean collection
OracleVectorStore.drop_collection(collection_name="ORACLE_KNOWLEDGE")

OracleVectorStore.create_collection(collection_name="ORACLE_KNOWLEDGE")

# create embedding model and then the vector store

# create the Embedding Model
embed_model = OCIGenAIEmbeddings(
    # this code is done to be run in OCI DS.
    # If outside replace with API_KEY and provide API_KEYS
    # auth_type = "RESOURCE_PRINCIPAL"
    auth_type="API_KEY",
    model_id=EMBED_MODEL,
    service_endpoint=ENDPOINT,
    compartment_id=COMPARTMENT_OCID,
)

# Here compute embeddings and load texts + embeddings in DB
# can take minutes (for embeddings)
v_store = OracleVectorStore.from_documents(
    docs_splitted, embed_model, collection_name="ORACLE_KNOWLEDGE", verbose=True
)

logging.info("")
logging.info("Loading correctly terminated!")

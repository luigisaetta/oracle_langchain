EMBEDDINGS_BITS = 64

# per ora usiamo il tokenizer di Cohere...
TOKENIZER = "Cohere/Cohere-embed-multilingual-v3.0"

#
# Some configurations fopr the load_vector_store
#

# directory where the Knowledge base is contained in txt files
TXT_DIR = "./txt"
# file with f_name, url
REF_FILE = "references.csv"

# OCI GenAI model used for Embeddings
EMBED_MODEL = "cohere.embed-multilingual-v3.0"
GENAI_MODEL = "cohere.command"
ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

# parameter for LLM
TEMPERATURE = 0.1
MAX_TOKENS = 1024

# number of docs returne dfrom the Retriever
TOP_K = 4

# max length in token of the input for embeddings
MAX_LENGTH = 512

# max chunk size, in char, for splitting
CHUNK_SIZE = 1500
# this parameters needs to be adjusted for the Embed model

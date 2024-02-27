import logging
from config import EMBED_MODEL, GENAI_MODEL, ENDPOINT, TOP_K


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def print_configuration():
    logging.info("------------------------")
    logging.info("Config. used:")
    logging.info(f"{EMBED_MODEL} for embeddings...")
    logging.info("Using Oracle DB Vector Store...")
    logging.info(f"Using {GENAI_MODEL} as LLM...")
    logging.info("")
    logging.info("Retrieval parameters:")
    logging.info(f"TOP_K: {TOP_K}")

    logging.info("------------------------")
    logging.info("")

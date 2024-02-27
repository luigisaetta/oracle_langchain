# Oracle AI Vector Search and GenAI with LangChain
This repository contains all the work done to develop demos based on LangChain, OCI Generative AI Service and Oracle AI Vector Search

## Data
In the txt subdirectory you can find a set of txt files, taken from Oracle docs and blogs, that you can use to setup quickly a demo.

## Setup the demo.
Clone the repository, setup your conda environemnt, have a ready DB with AI Vector Search and then:
* load the txt files + embeddings in the DB, using [load_vector_store](./load_vector_store.ipynb)
* query your knowledge base, using a [simple_assistant](./simple_assistant.ipynb) 

## Security (and Setup)
To connect to your Oracle DB (with AI Vector search) you need to prepoare a config_private.py file with user/pwd.

See details in the [Security section of the Wiki](https://github.com/luigisaetta/oracle_langchain/wiki/Security) 

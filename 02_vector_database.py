# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# Goals: Intro to ... 
# - Langchain Document Loaders
# - Text Embeddings
# - Vector Databases

# LIBRARIES 

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import pandas as pd
import pytimetk as tk
import yaml
import os

from pprint import pprint

# OPENAI API SETUP

os.environ['OPENAI_API_KEY'] = yaml.safe_load(open('credentials.yml'))['openai']

# 1.0 DATA PREPARATION ----

youtube_df = pd.read_csv('data/youtube_videos.csv')

youtube_df.glimpse()

# * Document Loaders
#   https://python.langchain.com/docs/integrations/document_loaders/pandas_dataframe 

loader = DataFrameLoader(youtube_df, page_content_column='page_content')

documents = loader.load()

documents[0]

documents[0].metadata
documents[0].page_content

pprint(documents[0].page_content)

for doc in documents:
    print(len(doc.page_content))

# * Text Splitting
#   https://python.langchain.com/docs/modules/data_connection/document_transformers

CHUNK_SIZE = 15000

# Character Splitter: Splits on simple default of 
text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=1000,
    separator=" "
)

docs = text_splitter.split_documents(documents)

pprint(docs[0].page_content)

len(docs)

# Recursive Character Splitter: Uses "smart" splitting, and recursively tries to split until text is small enough
text_splitter_recursive = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap=1000,
)

docs_recursive = text_splitter_recursive.split_documents(documents)

len(docs_recursive)


# * Post Processing Text- By default this transcript is literally words that the person is saying.
#That's good but it's not great when it comes to Youtube. 
#You want to inject it with a little bit of metadata. Things that i find important are title of video, author of video, URL as well. 


# IMPORTANT: Prepend the title and author to the page content
# - Helps with adding sources and searching titles
for doc in docs_recursive:
    # Retrieve the title and author from the document's metadata
    title = doc.metadata.get('title', 'Unknown Title')
    author = doc.metadata.get('author', 'Unknown Author')
    video_url = doc.metadata.get('video_url', 'Unknown URL')
    
    # Prepend the title and author to the page content
    updated_content = f"Title: {title}\nAuthor: {author}\nVideo URL: {video_url}\n\n{doc.page_content}"
    
    # Update the document's page content
    doc.page_content = updated_content

docs_recursive

pprint(docs_recursive[0].page_content)

# * Text Embeddings

# OpenAI Embeddings
# - See Account Limits for models: https://platform.openai.com/account/limits
# - See billing to add to your credit balance: https://platform.openai.com/account/billing/overview
#Embedding function allows you to create chroma DB. 

embedding_function = OpenAIEmbeddings(
    model='text-embedding-ada-002',
)

# Open Source Alternative:
# Requires Torch and SentenceTransformer packages:
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# * Langchain Vector Store: Chroma DB
# https://python.langchain.com/docs/integrations/vectorstores/chroma

# Creates a sqlite database called vector_store.db
# ONLY USE .from_documents() ONCE PER SESSION. Switch to Chroma() for next use once the db is created.
vectorstore = Chroma.from_documents(
    docs_recursive, 
    embedding=embedding_function, 
    persist_directory="data/chroma_1.db"
)

vectorstore

# Next time you run this code, it will load the database instead of recreating it
vectorstore = Chroma(
    embedding_function=embedding_function, 
    persist_directory="data/chroma_1.db"
)

vectorstore


# * Similarity Search: The whole reason we did this
#What documents are similar to that question? It will retrieve top 4 documents. 

result = vectorstore.similarity_search("How to create a social media strategy", k = 4)

result

pprint(result[0].page_content)
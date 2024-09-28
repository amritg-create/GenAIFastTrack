# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# Goals: Intro to ... 
# - Document Retrieval
# - Augmenting LLMs with the Expert Information

# LIBRARIES 

from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import yaml
import os

from pprint import pprint

# Key Parameters
#These are things you might need to change. For instance, if you change embedding model or database, this is useful to know. 
RAG_DATABASE    = "data/chroma_1.db"
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM             = "gpt-4o-mini"

# OPENAI_API_KEY

os.environ['OPENAI_API_KEY'] = yaml.safe_load(open('credentials.yml'))['openai']


# 1.0 CREATE A RETRIEVER FROM THE VECTORSTORE 
#We will use function chroma to connect up to vector store. 
#We have vector store that is like a database connection and it has as.retriever method. 

embedding_function = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
)

vectorstore = Chroma(
    persist_directory=RAG_DATABASE,
    embedding_function=embedding_function
)

retriever = vectorstore.as_retriever()

retriever

# 2.0 USE THE RETRIEVER TO AUGMENT AN LLM

# * Prompt template 

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# * LLM Specification

model = ChatOpenAI(
    model = LLM,
    temperature = 0.7,
)

# response = model.invoke("What are the top 3 things needed in a good social media marketing strategy for Facebook (Meta)? Site any sources used.")

# pprint(response.content)

# * Combine with Lang Chain Expression Language (LCEL)
#   - Context: Give it access to the retriever
#   - Question: Provide the user question as a pass through from the invoke method
#   - Use LCEL to add a prompt template, model spec, and output parsing
#Below we can use retriever and LLM model together. StrOutputParser is to make sure it produces string text. 

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# * Try it out:

# * Baseline
result_baseline = model.invoke("What are the top 3 things needed in a good social media marketing strategy for Facebook (Meta)? Site any sources used.")

pprint(result_baseline.content)

# * RAG
result = rag_chain.invoke("What are the top 3 things needed in a good social media marketing strategy for Facebook (Meta)? Site any sources used.")

pprint(result)



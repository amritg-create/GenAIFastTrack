# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# BONUS: HOW TO RUN LOCALL LLMS FOR DATA PRIVACY AND FREE LLM INFERENCE
# ***
# GOAL: Get Ollama set up to run local LLMs

# NOTE - This tutorial should be run AFTER completing the AI Fast Track because it depends on having LangChain installed


# STEP 1: ACTIVATE ds4b_301p ENVIRONMENT

# STEP 2: DOWNLOAD OLLAMA
#  https://www.ollama.com/download

# STEP 2A: PICK AN OPEN MODEL TO DOWNLOAD
#  https://www.ollama.com/library 

# STEP 3: INSTALL llama3 (NOTE: THE 8B MODEL IS 4.7GB AND TAKES 15+ MINUTES TO DOWNLOAD)
#ollama run llama3.1:8b

#   * Models are stored in ~/.ollama folder
#   * Ctrl + D to exit the interactive client

# STEP 4: INSTALL PYTHON OLLAMA PACKAGE
#  pip install ollama==0.2.0

# STEP 5: USE LANGCHAIN

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.1")
llm

response = llm.invoke("what color is the sky normally? Use one word response.")

response.content

# CONCLUDING COMMENTS:
# 1. WE ARE USING THE LLAMA3 8B PARAMETER MODEL, WHICH IS 4.7 GB
# 2. INFERENCE SPEED CAN BE SLOW FOR LONGER RESPONSES. 
# 3. SPEEDING UP INFERENCING TYPICALLY REQUIRES GPUS
# 4. CLOUD PLATFORMS LIKE AWS OFFER GPU-BASED SERVICES LIKE AMAZON BEDROCK (https://aws.amazon.com/bedrock/)

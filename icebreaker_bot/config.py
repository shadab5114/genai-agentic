"""Configuration settings for the Icebreaker Bot."""

import os
from dotenv import load_dotenv

# Load OPENAI_API_KEY (and any other secrets) from .env into the environment
load_dotenv()

# nltk's CWD import guard (added in nltk 3.10.1, see nltk/inisec.py) false-positives
# on this project's layout: venv/ lives inside icebreaker_bot/, so nltk's own
# dependencies (e.g. regex) resolve as "inside cwd" and get blocked when nltk
# imports them internally during SentenceSplitter's sentence tokenization.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

# OpenAI settings
# API key is read from the OPENAI_API_KEY environment variable (see .env), not stored here.
LLM_MODEL_ID = "gpt-4o-mini"
EMBEDDING_MODEL_ID = "text-embedding-3-small"

# ProxyCurl API settings
PROXYCURL_API_KEY = ""  # Replace with your API key

# Mock data URL
MOCK_DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"

# Query settings
SIMILARITY_TOP_K = 5
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 500
TOP_P = 1

# Node settings
CHUNK_SIZE = 500

# LLM prompt templates
INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's career or education.

Answer in detail, using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full details, using only the information provided in the context. If the answer is not available in the context, say "I don't know. The information is not available on the LinkedIn page."
"""

import os

from dotenv import load_dotenv  # For loading OPENAI_API_KEY out of .env
from langchain_openai import ChatOpenAI  # For interacting with OpenAI's chat models

# Read .env sitting next to this file so the script works from any directory
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY is not set. Add it to meeting_assistant/.env")

# Set the model ID
model_id = "gpt-4o-mini"

# Initialize the model
# OpenAI always samples, so there is no separate decoding-method setting;
# temperature and top_p are what shape the output
openai_llm = ChatOpenAI(
    model=model_id,
    temperature=0.5,
    top_p=1,
    max_tokens=1000,
)

response = openai_llm.invoke("How to read a book effectively?")
print(response.content)

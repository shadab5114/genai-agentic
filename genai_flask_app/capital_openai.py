from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    # api_key = "<YOUR_API_KEY>"  # or set OPENAI_API_KEY environment variable
)

params = {
    "temperature": 0,       # greedy-equivalent decoding
    "max_tokens": 100
}

text = """
Only reply with the answer. What is the capital of Canada?
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": text}],
    **params
)

print(response.choices[0].message.content)

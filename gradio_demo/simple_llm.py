# Import the necessary packages
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import gradio as gr

# Load OPENAI_API_KEY from .env
load_dotenv()

# Specify the model
model_id = "gpt-4o-mini"  # Swap for "gpt-4o" if you want stronger answers at higher cost

# Wrap up the model into a LangChain chat model
llm = ChatOpenAI(
    model=model_id,
    max_tokens=512,   # Specify the max tokens you want to generate
    temperature=0.5,  # The randomness or creativity of the model's responses
)

# Get the query from the user input
# query = input("Please enter your query: ")

# Print the generated response
# print(llm.invoke(query).content)

# Function to generate a response from the model
def generate_response(prompt_txt):
    response = llm.invoke(prompt_txt)
    return response.content

# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    flagging_mode="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Gradio Chat with LLM",
    description="This is a simple chat application using Gradio and LangChain with an OpenAI model.",
)

# Launch the Gradio interface
chat_application.launch(server_name="127.0.0.1", server_port=7860)
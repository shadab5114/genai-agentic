import os
import re

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import LLMChain, RetrievalQA

from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS

import gradio as gr

def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(url):
    # Extracts the video ID from the URL
    video_id = get_video_id(url)
    
    # Create a YouTubeTranscriptApi() object
    ytt_api = YouTubeTranscriptApi()
    
    # Fetch the list of available transcripts for the given YouTube video
    transcripts = ytt_api.list(video_id)
    
    transcript = ""
    for t in transcripts:
        # Check if the transcript's language is English
        if t.language_code == 'en':
            if t.is_generated:
                # If no transcript has been set yet, use the auto-generated one
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                # If a manually created transcript is found, use it (overrides auto-generated)
                transcript = t.fetch()
                break  # Prioritize the manually created transcript, exit the loop
    
    return transcript if transcript else None

# Sample YouTube URL
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Fetching the transcript
transcript = get_transcript(url)

# Output the fetched transcript
print(transcript)

def process(transcript):
    # Initialize an empty string to hold the formatted transcript
    txt = ""
    
    # Loop through each entry in the transcript
    for i in transcript:
        try:
            # Append the text and its start time to the output string
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass
            
    # Return the processed transcript as a single string
    return txt

# Processing the transcript
formatted_transcript = process(transcript)

# Output the processed transcript
print(formatted_transcript)

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks

# Chunking the transcript
chunks = chunk_transcript(formatted_transcript)

# Output the chunks
print(chunks)


def setup_credentials():
    # Load environment variables (OPENAI_API_KEY) from the .env file
    load_dotenv()

    # Define the model ID for the OpenAI model being used
    model_id = "gpt-4o-mini"

    # Retrieve the OpenAI API key from the environment
    api_key = os.getenv("OPENAI_API_KEY")

    # Return the model ID and API key for later use
    return model_id, api_key


def define_parameters():
    # Return a dictionary containing the parameters for the OpenAI model
    return {
        # Temperature of 0 mirrors GREEDY decoding: deterministic output
        "temperature": 0,

        # Specify the maximum number of new tokens to generate
        "max_tokens": 900,
    }


def initialize_openai_llm(model_id, api_key, parameters):
    # Create and return an instance of ChatOpenAI with the specified configuration
    return ChatOpenAI(
        model=model_id,          # Set the model ID for the LLM
        api_key=api_key,         # Pass the OpenAI API key
        **parameters             # Pass the parameters for model behavior
    )

def setup_embedding_model(api_key):
    # Create and return an instance of OpenAIEmbeddings with the specified configuration
    return OpenAIEmbeddings(
        model='text-embedding-3-small',  # Set the model ID for the embedding model
        api_key=api_key                   # Pass the OpenAI API key
    )

def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded transcript using the FAISS index.

    :param faiss_index: The FAISS index containing embedded text chunks
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    :return: List of similar results
    """
    # Perform the similarity search using the FAISS index
    results = faiss_index.similarity_search(query, k=k)
    return results

def create_summary_prompt():
    """
    Create a ChatPromptTemplate for summarizing a YouTube video transcript.

    :return: ChatPromptTemplate object
    """
    # Define the system and human messages for the summary prompt
    system_message = (
        "You are an AI assistant tasked with summarizing YouTube video transcripts. "
        "Provide concise, informative summaries that capture the main points of the video content.\n\n"
        "Instructions:\n"
        "1. Summarize the transcript in a single concise paragraph.\n"
        "2. Ignore any timestamps in your summary.\n"
        "3. Focus on the spoken content (Text) of the video.\n\n"
        "Note: In the transcript, \"Text\" refers to the spoken words in the video, "
        "and \"start\" indicates the timestamp when that part begins in the video."
    )
    human_message = "Please summarize the following YouTube video transcript:\n\n{transcript}"

    # Create the ChatPromptTemplate object with the defined messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message),
    ])

    return prompt

def create_summary_chain(llm, prompt, verbose=True):
    """
    Create an LLMChain for generating summaries.

    :param llm: Language model instance
    :param prompt: ChatPromptTemplate instance
    :param verbose: Boolean to enable verbose output (default: True)
    :return: LLMChain instance
    """
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)

def create_faiss_index(chunks, embedding_model):
    """
    Build a FAISS index from transcript chunks using the given embedding model.

    Args:
        chunks: list[str]
            The transcript text chunks to embed and index.
        embedding_model: OpenAIEmbeddings
            The embedding model used to vectorize the chunks.

    Returns:
        FAISS: A FAISS index built from the embedded chunks.
    """
    return FAISS.from_texts(chunks, embedding_model)

def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.

    Parameters:
        query (str): The user's query string.
        faiss_index (FAISS): The FAISS index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 3).

    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context



def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.

    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """
    
    # Define the template string
    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.

    Relevant Video Context: {context}

    Based on the above context, please answer the following question:
    Question: {question}
    """

    # Create the PromptTemplate object
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )

    return prompt_template


# Creating the Q&A prompt template 
qa_prompt_template = create_qa_prompt_template()

# Example of how to use the prompt template with context and a question
context = "This video explains the fundamentals of quantum physics."
question = "What are the key principles discussed in the video?"

# Generating the prompt
generated_prompt = qa_prompt_template.format(context=context, question=question)

# Output the generated prompt
print(generated_prompt)

def create_qa_chain(llm, prompt_template, verbose=True):
    """
    Create an LLMChain for question answering.

    Args:
        llm: Language model instance
            The language model to use in the chain (e.g., WatsonxGranite).
        prompt_template: PromptTemplate
            The prompt template to use for structuring inputs to the language model.
        verbose: bool, optional (default=True)
            Whether to enable verbose output for the chain.

    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """
    
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)


def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain: LLMChain
            The question-answering chain (LLMChain) to use for generating answers.
        k: int, optional (default=3)
            The number of relevant documents to retrieve.

    Returns:
        str: The generated answer to the user's question.
    """

    # Retrieve relevant context
    relevant_context = retrieve(question, faiss_index, k=k)

    # Join the retrieved document chunks into a single context string
    context_text = "\n\n".join(doc.page_content for doc in relevant_context)

    # Generate answer using the QA chain
    answer = qa_chain.predict(context=context_text, question=question)

    return answer

# Initialize an empty string to store the processed transcript after fetching and preprocessing
processed_transcript = ""

def summarize_video(video_url):
    """
    Title: Summarize Video

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript

    if video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:
        # Step 1: Set up OpenAI credentials
        model_id, api_key = setup_credentials()

        # Step 2: Initialize OpenAI LLM for summarization
        llm = initialize_openai_llm(model_id, api_key, define_parameters())

        # Step 3: Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # Step 4: Generate the video summary
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."


def answer_question(video_url, user_question):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global fetched_transcript, processed_transcript

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)

        # Step 2: Set up OpenAI credentials
        model_id, api_key = setup_credentials()

        # Step 3: Initialize OpenAI LLM for Q&A
        llm = initialize_openai_llm(model_id, api_key, define_parameters())

        # Step 4: Create FAISS index for transcript chunks (only needed for Q&A)
        embedding_model = setup_embedding_model(api_key)
        faiss_index = create_faiss_index(chunks, embedding_model)

        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 6: Generate the answer using FAISS index
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."


with gr.Blocks() as interface:
    # Input field for YouTube URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
    
    # Outputs for summary and answer
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

    # Buttons for selecting functionalities after fetching transcript
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    # Set up button actions
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

# Launch the app with specified server name and port
interface.launch(server_name="127.0.0.1", server_port=7860)

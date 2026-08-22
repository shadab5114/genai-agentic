import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from transformers import pipeline  # For Speech-to-Text

from ffmpeg_setup import ensure_ffmpeg

# The transformers ASR pipeline decodes audio by shelling out to ffmpeg
ensure_ffmpeg()

#######------------- LLM Initialization-------------#######

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY is not set. Add it to meeting_assistant/.env")

model_id = "gpt-4o-mini"

# Initialize the OpenAI LLM used to write the meeting minutes
llm = ChatOpenAI(
    model=model_id,
    temperature=0.2,
    top_p=1,
    max_tokens=1000,
)

# A second, more literal model for the terminology clean-up pass.
# Low temperature because we want faithful edits, not creative rewriting.
terminology_llm = ChatOpenAI(
    model=model_id,
    temperature=0.2,
    top_p=0.6,
    max_tokens=2000,
)

#######------------- Speech-to-Text Model-------------#######

# Loaded once at import time. Rebuilding this per request would re-read the
# model weights from disk on every upload.
ASR_MODEL = "openai/whisper-tiny.en"

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=ASR_MODEL,
)

#######------------- Helper Functions-------------#######

# Function to remove non-ASCII characters
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)


def product_assistant(ascii_transcript):
    system_prompt = """You are an intelligent assistant specializing in financial products;
    your task is to process transcripts of earnings calls, ensuring that all references to
    financial products and common financial terms are in the correct format. For each
    financial product or common term that is typically abbreviated as an acronym, the full term
    should be spelled out followed by the acronym in parentheses. For example, '401k' should be
    transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 'Health Savings Account (HSA)' , 'ROA' should be transformed to 'Return on Assets (ROA)', 'VaR' should be transformed to 'Value at Risk (VaR)', and 'PB' should be transformed to 'Price to Book (PB) ratio'. Similarly, transform spoken numbers representing financial products into their numeric representations, followed by the full name of the product in parentheses. For instance, 'five two nine' to '529 (Education Savings Plan)' and 'four zero one k' to '401(k) (Retirement Savings Plan)'. However, be aware that some acronyms can have different meanings based on the context (e.g., 'LTV' can stand for 'Loan to Value' or 'Lifetime Value'). You will need to discern from the context which term is being referred to and apply the appropriate transformation. In cases where numerical figures or metrics are spelled out but do not represent specific financial products (like 'twenty three percent'), these should be left as is. Your role is to analyze and adjust financial product terminology in the text. Once you've done that, produce the adjusted transcript and a list of the words you've changed"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ascii_transcript},
    ]

    response = terminology_llm.invoke(messages)
    return response.content

#######------------- Prompt Template and Chain-------------#######

# Define the prompt template
template = """
You are writing meeting minutes from a transcript. Use ONLY information
explicitly stated in the transcript below. Do not infer, embellish, or invent
decisions, tasks, names, dates, or figures. If a section has no supporting
content in the transcript, write "None stated." and move on. Never emit
placeholders like [Insert Date] or [Name].

Transcript:
{context}

Produce:

Key points discussed:
- ...

Decisions made:
- Only decisions actually announced in the transcript.

Task list:
- Only action items actually assigned. Include an assignee or deadline only if
  the transcript names one; otherwise omit that detail rather than guessing.
"""


prompt = ChatPromptTemplate.from_template(template)

# Define the chain: fill the template, send it to the LLM, keep the text reply
chain = prompt | llm | StrOutputParser()

#######------------- Speech2text and Pipeline-------------#######

# Speech-to-text pipeline
def transcript_audio(audio_file):
    # Gradio hands us the path of the uploaded file, or None if nothing was attached
    if not audio_file:
        return "Please upload an audio file first.", None

    # `return_timestamps=True` enables Whisper's own long-form algorithm, which
    # stitches its 30-second windows together so nothing is dropped at the seams
    raw_transcript = asr_pipe(audio_file, return_timestamps=True)["text"]
    ascii_transcript = remove_non_ascii(raw_transcript)

    adjusted_transcript = product_assistant(ascii_transcript)
    result = chain.invoke({"context": adjusted_transcript})

    # Write the result to a file for downloading
    output_file = Path(__file__).parent / "meeting_minutes_and_tasks.txt"
    output_file.write_text(result, encoding="utf-8")

    # Return the textual result and the file for download
    return result, str(output_file)

#######------------- Gradio Interface-------------#######

audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload your audio file")
output_text = gr.Textbox(label="Meeting Minutes and Tasks")
download_file = gr.File(label="Download the Generated Meeting Minutes and Tasks")

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=[output_text, download_file],
    title="AI Meeting Assistant",
    description="Upload an audio file of a meeting. This tool will transcribe the audio, fix product-related terminology, and generate meeting minutes along with a list of tasks."
)

iface.launch(server_name="127.0.0.1", server_port=5000)

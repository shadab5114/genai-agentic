from transformers import pipeline
import gradio as gr

from ffmpeg_setup import ensure_ffmpeg

ensure_ffmpeg()

# Initialize the speech-to-text pipeline from Hugging Face Transformers
# This uses the "openai/whisper-tiny.en" model for automatic speech recognition (ASR)
# Built once at import time rather than per request, so the model loads only once
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
)


# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    # Gradio hands us the path of the uploaded file, or None if nothing was attached
    if not audio_file:
        return "Please upload an audio file first."

    # Perform speech recognition on the uploaded file
    # Whisper only hears 30 seconds at a time. `return_timestamps=True` switches on
    # its own long-form algorithm, which stitches those windows back together with
    # overlap so nothing is dropped at the seams.
    # The result is stored with the key "text" containing the transcribed text
    result = pipe(audio_file, return_timestamps=True)["text"]

    return result.strip()


# Set up Gradio interface
audio_input = gr.Audio(sources=["upload"], type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn=transcript_audio,
                     inputs=audio_input, outputs=output_text,
                     title="Audio Transcription App",
                     description="Upload the audio file")

# Launch the Gradio app
iface.launch(server_name="127.0.0.1", server_port=5000)

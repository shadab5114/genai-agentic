from pathlib import Path

from transformers import pipeline

from ffmpeg_setup import ensure_ffmpeg


ensure_ffmpeg()

# Initialize the speech-to-text pipeline from Hugging Face Transformers
# This uses the "openai/whisper-tiny.en" model for automatic speech recognition (ASR)
pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
)

# Define the path to the audio file that needs to be transcribed
# Resolved relative to this file so the script runs from any directory
sample = Path(__file__).parent / 'sample-meeting.wav'

# Perform speech recognition on the audio file
# Whisper only hears 30 seconds at a time. `return_timestamps=True` switches on
# its own long-form algorithm, which stitches those windows back together with
# overlap so nothing is dropped at the seams.
# The result is stored in `prediction` with the key "text" containing the transcribed text
prediction = pipe(str(sample), return_timestamps=True)["text"]

# Print the transcribed text to the console
print(prediction.strip())

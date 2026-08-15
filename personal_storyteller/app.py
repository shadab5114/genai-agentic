import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_ID = "gpt-4o-mini"
TTS_MODEL_ID = "gpt-4o-mini-tts"
TTS_VOICE = "nova"
TTS_INSTRUCTIONS = (
    "Voice: an 8-year-old girl telling her best friend an amazing story. "
    "Tone: bubbly, excited, and enthusiastic, like she can barely contain "
    "how cool this is. Pitch: light and high, youthful. Pacing: quick and "
    "energetic, with little breathless bursts of excitement. Delivery: "
    "big emphasis and a giggly grin on fun facts and the reveal moments, "
    "like she's sharing the best secret ever."
    "talks fast like she just drank juice"
)


def generate_story(topic: str) -> str:
    prompt = f"""Write an engaging and educational story about {topic} for beginners.
Use simple and clear language to explain basic concepts.
Include interesting facts and keep it friendly and encouraging.
The story should be around 200-300 words and end with a brief summary of what we learned.
Make it perfect for someone just starting to learn about this topic."""

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def narrate_story(story: str, output_path: str = "generated_story.mp3") -> str:
    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL_ID,
        voice=TTS_VOICE,
        input=story,
        instructions=TTS_INSTRUCTIONS,
    ) as response:
        response.stream_to_file(output_path)
    return output_path


if __name__ == "__main__":
    topic = "the life cycle of butterflies"

    story = generate_story(topic)
    print("Generated Story:\n", story)

    audio_path = narrate_story(story)
    print(f"\nAudio saved to: {audio_path}")

# Image Captioning with OpenAI Vision

Ask a vision model questions about an image. Same structure as the watsonx
lab, with OpenAI's models in place of IBM's Granite / Llama Vision.

```
image URL  ->  requests.get  ->  base64  ->  data URI
                                                |
                                    chat.completions.create
                                    (text part + image part)
                                                |
                                          model's answer
```

---

## What it does

One script, `image_captioning.py`, runs four kinds of visual query against four
images:

| Section | Image | Question |
|---|---|---|
| Image captioning | all four | "Describe the photo" |
| Object detection | runner on a road | "How many cars are in this image?" |
| Damage assessment | aerial view of a flooded farm | "How severe is the damage in this image?" |
| Label extraction | Nutrition Facts panel | "How much sodium is in this product?" |

---

## The one idea worth remembering

A model can't open a `.png`. The pixels travel inside the request body, as a
base64 data URI, and a user message carries them by making `content` a **list of
parts** instead of a plain string:

```python
"content": [
    {"type": "text",      "text": "How many cars are in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
]
```

Everything else - roles, `temperature`, `max_tokens` - works exactly as it does
for text-only calls.

---

## Setup

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then create `.env` in this folder (see `.env.example`):

```
OPENAI_API_KEY=sk-...
```

## Run

```powershell
python image_captioning.py
```

---

## Notes

**Model choice.** `model_id` at the top of the script is the only thing to change
to swap models. `gpt-4o-mini` is the cheap default and handles all four tasks;
`gpt-4o` is meaningfully better on small print - on the nutrition label it reads
every % Daily Value correctly, where `gpt-4o-mini` misreads one of them.

**Image size.** These samples are large (image-1 is 8 MB) and base64 adds about a
third on top. It works as-is, but if you hit request-size limits or want cheaper
calls, downscale with Pillow to ~1536px on the long edge before encoding - the
vision endpoint scales images down to roughly that anyway.

**Your own images.** Add URLs to `image_urls`, or read a local file directly:

```python
encoded_image = base64.b64encode(open("photo.png", "rb").read()).decode("utf-8")
```

# AI Meeting Assistant

Turn a meeting recording into structured minutes and an action-item list.

Audio goes in one end, and a grounded summary comes out the other:

```
audio file  ->  Whisper (local)  ->  raw transcript
                                          |
                                     ASCII clean-up
                                          |
                              GPT terminology pass  ->  corrected transcript
                                          |
                                  GPT summarizer  ->  minutes + task list
                                          |
                              Gradio UI + downloadable .txt
```

Two stages run locally on your machine (speech recognition), and two are OpenAI
API calls (terminology correction and summarization).

---

## What was accomplished

**Text generation with LLMs.** Built a script that generates text through
OpenAI's chat models via LangChain, and explored the parameters that shape the
output — `temperature`, `top_p`, and `max_tokens` — including how to swap models
by changing a single identifier.

**Speech-to-text conversion.** Used OpenAI's Whisper model, running locally
through Hugging Face `transformers`, to transcribe a 56-second earnings-call
recording. Along the way, worked out why the naive configuration silently
discarded the first 7.5 seconds of every recording, and fixed it.

**Content summarization.** Chained two GPT calls: one that expands financial
jargon into its correct written form ("HSA" becomes "Health Savings Account
(HSA)"), and one that extracts key points, decisions, and action items. Tuned
the prompt so the model reports "None stated." rather than inventing decisions
that were never made.

**User interface development.** Wrapped the pipeline in a Gradio web interface
with file upload, a results pane, and a download link — no command line needed.

---

## Files

| File | What it does |
|---|---|
| `speech_analyzer.py` | **The full application.** Audio to minutes, with the Gradio UI. |
| `simple_llm.py` | Minimal OpenAI text generation — the "hello world" for the LLM layer. |
| `simple_text2speech.py` | Minimal command-line transcription of `sample-meeting.wav`. |
| `speech2Text_app.py` | Transcription only, with a Gradio upload interface. |
| `ffmpeg_setup.py` | Shared helper that guarantees ffmpeg is reachable (see below). |
| `sample-meeting.wav` | Test recording — 56s of a fictional earnings call. |
| `requirements.txt` | Python dependencies. |

---

## Setup

**1. Python packages** (a virtualenv already exists in `venv/`):

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**2. ffmpeg** — Whisper cannot decode audio files without it:

```powershell
winget install Gyan.FFmpeg
```

**3. Your OpenAI key** — create or edit `.env` in this folder:

```
OPENAI_API_KEY=sk-...
```

## Running

```powershell
.\venv\Scripts\python.exe speech_analyzer.py
```

Then open http://127.0.0.1:5000 and upload an audio file.

To try just the transcription step, without needing an API key:

```powershell
.\venv\Scripts\python.exe simple_text2speech.py
```

---

## Versions

Built against these, all current as of the build:

| Package | Version |
|---|---|
| transformers | 5.15.0 |
| torch | 2.13.0+cpu |
| gradio | 6.24.0 |
| langchain | 1.3.15 |
| langchain-openai | 1.5.1 |
| openai | 3.1.0 |
| python-dotenv | 1.2.3 |
| ffmpeg | 9.0 |

Note that LangChain 1.x and transformers 5.x are major releases with breaking
changes. Tutorial code written for LangChain 0.3 / transformers 4.x — anything
using `LLMChain`, `initialize_agent`, or `chunk_length_s` — needs adapting.

---

## Problems solved along the way

These were real bugs, and each one is worth remembering.

### Whisper silently dropped the first 7.5 seconds

Setting `chunk_length_s=30` invokes the generic transformers chunker, which
loses audio at the window boundaries. It fails **silently** — you get a
plausible transcript with a sentence missing and no error.

The fix is `return_timestamps=True`, which switches on Whisper's own long-form
algorithm. That one uses overlapping windows and timestamp-based stitching, so
nothing falls through the seams.

```python
# Wrong - drops audio
pipe = pipeline("automatic-speech-recognition", model=..., chunk_length_s=30)
result = pipe(audio, batch_size=8)["text"]

# Right
pipe = pipeline("automatic-speech-recognition", model=...)
result = pipe(audio, return_timestamps=True)["text"]
```

### The summarizer invented decisions that never happened

The original prompt listed "Decisions made" and "Actionable items with
assignees and deadlines" as headings to fill in. Given a transcript containing
neither, GPT manufactured both — including an approved communications strategy
that nobody had approved, and a task table of invented assignments.

For a meeting assistant this is the worst failure mode available: fabricated
action items are indistinguishable from real ones, and somebody downstream acts
on them.

Two changes fixed it: instruct the model to use only what is explicitly stated
and to write "None stated." for empty sections, and drop `temperature` from 0.5
to 0.2. Summarization wants faithfulness, not variety.

### ffmpeg was installed but invisible to Python

```
ValueError: ffmpeg was not found but is required to load audio files
```

On Windows, winget updates the persistent PATH, but processes already running —
VS Code and every terminal it spawned — keep their old copy of the environment
until they restart. `ffmpeg_setup.py` sidesteps this entirely by reading the
real PATH from the registry at import time, so the code works from a terminal,
the VS Code Run button, or a notebook without any restart.

### Smaller things

- The model was being rebuilt inside the request handler, reloading its weights
  from disk on **every** upload. Moved to module level.
- `{"context": RunnablePassthrough()} | prompt` invoked with a dict made
  `context` the whole dictionary, so the prompt contained a literal
  `{'context': '...'}`. Simplified to `prompt | llm | StrOutputParser()`.
- `open(path, "w")` on Windows defaults to cp1252 and crashes on an em dash.
  Always pass `encoding="utf-8"`.
- Gradio 6 wants `sources=["upload"]` — a list, not a string.
- `server_name="0.0.0.0"` exposes the app to your whole network.
  `127.0.0.1` keeps it local.

---

## Tuning

**Transcription accuracy vs. speed.** Change `ASR_MODEL` in
`speech_analyzer.py`:

| Model | Size | 56s clip on CPU | Quality |
|---|---|---|---|
| `openai/whisper-tiny.en` | 39M | ~16s | Rough — fumbles proper nouns and numbers |
| `openai/whisper-base.en` | 74M | ~30s | Noticeably better |
| `openai/whisper-medium` | 769M | several minutes | Good on financial jargon |

torch here is the **CPU** build, so larger models are slow. A CUDA build would
change these numbers dramatically.

Worth knowing: the GPT terminology pass repairs some transcription errors on
its own, so `tiny.en` plus the correction step can rival a much larger model at
a fraction of the compute. Test before paying for `medium`.

**Summarization behavior.** The prompt template and `temperature` in
`speech_analyzer.py` control how conservative the minutes are.

---

## Known limitations

- **Only the negative case is proven.** The sample recording contains no
  decisions or tasks, so we have confirmed the model won't *invent* them. It has
  not been verified that it still *captures* them when they are real — an
  over-corrected prompt can start reporting "None stated." for genuine action
  items. Test with a recording where someone actually assigns work.
- `whisper-tiny.en` is English-only. Use the non-`.en` models for other
  languages.
- No speaker diarization — the transcript does not identify who said what, so
  minutes cannot attribute statements to individuals.
- The whole recording is sent to OpenAI in a single request. Very long meetings
  will exceed the context window and need chunking.
- `app.py` is currently an empty placeholder.

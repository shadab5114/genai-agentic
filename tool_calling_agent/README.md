# YouTube Tool-Calling Agent

Give an LLM a set of Python functions ("tools") that can search YouTube,
pull video metadata, fetch transcripts, and grab thumbnails — then let the
model decide, on its own, which tools to call and in what order to answer a
question like *"summarize this video"* or *"show me the top 3 trending
videos in the US with metadata and thumbnails."*

The script builds up the same problem three times, each version more
capable than the last: a fully manual tool-calling walkthrough, an
automated fixed-length chain, and finally a recursive chain that can take
as many tool-calling rounds as the task actually needs.

## App overview

### 1. Custom tools

Five `@tool`-decorated functions extend what the LLM can do, each wrapping
a plain Python function with the name/description/schema LangChain needs to
expose it to the model:

| Tool | Library | Purpose |
|---|---|---|
| `extract_video_id` | `re` | Pull the 11-character video ID out of any YouTube URL format (`watch?v=`, `youtu.be/`, `embed/`) |
| `fetch_transcript` | `youtube-transcript-api` | Fetch a video's transcript by ID and language code |
| `search_youtube` | `pytube.Search` | Search YouTube by keyword, returning title/ID/URL for each hit |
| `get_full_metadata` | `yt-dlp` | Pull title, views, duration, channel, likes, comments, chapters for a video URL |
| `get_thumbnails` | `yt-dlp` | List available thumbnail URLs/resolutions for a video URL |

Each tool has a docstring that becomes the tool's `description` — this is
what the LLM actually reads to decide *when* to call it, so the docstrings
are written as instructions to the model, not just comments for humans.
All five are collected into a `tools` list and bound to the model with
`llm.bind_tools(tools)`, which tells OpenAI's API about their names, JSON
schemas, and descriptions.

### 2. Manual tool calling (the mechanics)

Before automating anything, the script walks through one full round trip
by hand to show what LangChain is actually doing under the hood:

1. Send the user's question to `llm_with_tools`.
2. The model doesn't answer directly — it replies with a `tool_calls` list
   instead (e.g. "call `extract_video_id` with this URL").
3. The app runs that tool itself and wraps the result in a `ToolMessage`,
   tagged with the same `tool_call_id` the model used.
4. That `ToolMessage` gets appended to the conversation, and the model is
   invoked again — now it can see the tool's output and either calls
   another tool (e.g. `fetch_transcript` once it has the video ID) or
   answers in plain text.

This surfaces a hard requirement of the OpenAI API: **an assistant message
containing `tool_calls` must be immediately followed by a `ToolMessage` for
every one of those calls** before you can invoke the model again — skip
that step and the API rejects the request with a 400 error.

### 3. Automated fixed-length chains

The manual version is turned into a reusable LangChain **Runnable**
pipeline (`summarization_chain`, later decomposed into `chain`) using
`RunnablePassthrough.assign(...)` to thread state through each step:
first LLM call → execute its tool calls → second LLM call → execute those
tool calls → final LLM call to produce the answer.

This works well for the video-summarization case, which always takes
exactly two tool rounds (`extract_video_id` then `fetch_transcript`). It
breaks down for anything else: a query like *"top 3 trending videos with
metadata and thumbnails"* needs a variable number of rounds (search, then
metadata, then thumbnails, possibly for several videos at once), and a
chain hardcoded to two rounds simply stops before the model is done.

### 4. Recursive tool calling (the general solution)

The final version, `universal_chain`, removes the round limit entirely:

```
ask the model
  -> did it request tool calls?
       yes -> run them, feed results back, ask again (recurse)
       no  -> done, return the conversation
```

`_recursive_chain` implements exactly that loop as plain recursion,
`should_continue` is the stopping condition (true as long as the last
message has `tool_calls`), and `process_tool_calls` executes every tool
call in the latest response before invoking the model again. Because this
places no cap on the number of rounds, it handles both the simple
two-step summarization case and open-ended, multi-tool queries the fixed
chains couldn't — the recursion naturally stops the moment the model
returns a plain-text answer instead of another tool call.

## Tech stack

| Layer | Choice |
|---|---|
| LLM | OpenAI `gpt-4o-mini` via `langchain.chat_models.init_chat_model` |
| Tool framework | LangChain (`langchain-core`, `langchain-community`, `langchain-openai`) |
| Orchestration | LangChain Runnables (`RunnablePassthrough`, `RunnableLambda`) |
| YouTube search | `pytube` |
| YouTube metadata/thumbnails | `yt-dlp` |
| YouTube transcripts | `youtube-transcript-api` |
| Secrets | `python-dotenv` (loads `OPENAI_API_KEY` from `.env`) |

## Project structure

```
tool_calling_agent/
|-- app.py              # Tools, manual chain, automated chain, recursive chain
|-- requirements.txt
|-- .env.example         # Copy to .env and add OPENAI_API_KEY
```

## Setup

1. Activate the virtual environment (already set up in `venv/`).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. `app.py` loads
   it automatically via `python-dotenv`.
4. Run the script:
   ```
   python app.py
   ```
   It runs through all three stages (manual, automated, recursive) against
   sample queries and prints each result.

## Known limitations

- `pytube.Search` depends on scraping YouTube's frontend and can break or
  emit noisy warnings whenever YouTube changes its page structure — the
  `pytube` logger is set to `ERROR` to suppress that noise, but the
  underlying fragility remains. `yt-dlp` (also used here for metadata and
  thumbnails) is the more actively maintained alternative if `Search`
  starts failing outright.
- `youtube-transcript-api` fetches transcripts by scraping YouTube
  directly, which YouTube will occasionally rate-limit or block per-IP,
  especially under repeated rapid requests. `fetch_transcript` always
  returns an `"Error: ..."` string rather than raising, so this shows up
  as the LLM explaining it couldn't get a transcript rather than a crash.
- The recursive chain has no maximum-iteration cap. In practice the model
  reliably stops once it has enough information to answer in plain text,
  but there's no hard ceiling guarding against a model that keeps
  requesting tools indefinitely.

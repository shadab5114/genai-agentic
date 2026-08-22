# Style Finder - Multimodal RAG for Fashion Analysis

Upload a fashion photo. The app identifies the garments, retrieves the real
item details behind them - names, prices, purchase links - and writes a styling
analysis grounded in that retrieved data.

The dataset is built from Taylor Swift's outfits, which makes it a clean
demonstration of the actual point: connecting a **visual** input to **structured**
commercial data, so the model talks about specific real garments instead of
guessing from pixels alone.

```
                        user uploads a photo
                                 |
                                 v
              +--------------------------------------+
              |   1. MULTIMODAL INPUT PROCESSING     |
              |   PIL -> 224x224 -> normalize        |
              |   ResNet50 -> feature vector         |--+
              +--------------------------------------+  |
                                 |                      |
                     1000-dim float32 vector            | base64 of the
                                 |                      | same image
                                 v                      |
              +--------------------------------------+  |
              |   2. VECTOR-BASED RETRIEVAL          |  |
              |   cosine similarity vs. every        |<-+-- swift-style-embeddings.pkl
              |   pre-encoded outfit                 |      192 items / 93 outfits
              |   -> closest outfit + score          |      (embeddings precomputed)
              +--------------------------------------+  |
                                 |                      |
              Item Name / Price / Link for that outfit   |
                     (the retrieved context)             |
                                 |                      |
                                 v                      v
              +--------------------------------------+
              |   3. CONTEXT-ENHANCED GENERATION     |
              |   prompt = retrieved context         |
              |          + the image itself          |
              |   -> vision LLM                      |
              +--------------------------------------+
                                 |
                                 v
                    grounded fashion analysis
```

---

## The multimodal RAG pipeline

### 1. Multimodal input processing

The uploaded image is processed and converted into a vector representation that
captures its essential visual features. The same image is base64-encoded in
parallel, because stage 3 needs the actual pixels, not just the vector.

Handled by `ImageProcessor.encode_image()`, which returns both halves at once:

```python
{"base64": "...", "vector": array([...], dtype=float32)}
```

### 2. Vector-based retrieval

The input vector is compared against a database of pre-encoded fashion images.
**Cosine similarity** finds the closest match in that vector space, and the
structured data attached to the winning image - item names, prices, purchase
links - is pulled out as context.

Retrieval is a two-step join:

| Step | Function | Result |
|---|---|---|
| Nearest neighbour | `ImageProcessor.find_closest_match()` | one row + a similarity score |
| Expand to full outfit | `get_all_items_for_image()` | every row sharing that `Image URL` |

The second step matters: rows are *items*, not outfits. One matched photo can
carry up to 11 garments and accessories, and all of them belong in the context.

The similarity score then steers the prompt, at the `SIMILARITY_THRESHOLD = 0.8`
set in `config.py`:

- **at or above 0.8** - treated as the same outfit; the model describes these
  exact items with their real prices and links.
- **below 0.8** - treated as merely similar; the model describes what it sees
  and offers the retrieved items as comparable alternatives.

### 3. Context-enhanced generation

The retrieved fashion data is formatted into a prompt for the vision LLM, and
sent together with the image. The model generates its response from **both** the
visual input and the retrieved context.

That augmentation is the whole value: a vision model alone can say "a tweed
blazer." With retrieval attached it can say *Versace 'Tweed Masculine Blazer',
$3,350.00*, and link to it - accuracy and detail the image alone cannot support.

---

## `models/image_processor.py`

The module behind stages 1 and 2. It uses a pre-trained neural network
(**ResNet50**) to turn images into vectors, so that two images can be compared
mathematically rather than pixel by pixel.

### `__init__`

Three attributes get set up once, at construction, and reused for every upload:

| Attribute | What it holds |
|---|---|
| `self.device` | `cuda` if a GPU is available, otherwise `cpu` |
| `self.model` | pre-trained ResNet50, moved to `self.device`, in **evaluation mode** |
| `self.preprocess` | a `transforms.Compose` pipeline: resize -> to tensor -> normalize |

Evaluation mode matters: it switches off dropout and freezes the batch-norm
running statistics, so the same image always produces the same vector.

The preprocess pipeline resizes to `IMAGE_SIZE` and normalizes with
`NORMALIZATION_MEAN` / `NORMALIZATION_STD` from `config.py` - the standard
ImageNet statistics ResNet50 was trained on. Feeding it un-normalized pixels
produces vectors that are quietly wrong rather than obviously broken.

### `encode_image(image_input, is_url=True)`

Converts one image into the two things the rest of the pipeline needs: a
**base64 string** for the LLM, and a **feature vector** for similarity search.

1. Loads the image - downloading it when `is_url=True`, reading from the local
   filesystem otherwise - and converts to **RGB** for consistency.
2. Encodes it to a base64 JPEG string through an in-memory `BytesIO` buffer,
   which is how the pixels travel to a vision-language model.
3. Runs `self.preprocess` and shapes the result into a single-image batch on
   `self.device`.
4. Passes that tensor through ResNet50 to extract high-level features.
5. Flattens the output to a one-dimensional NumPy array - the embedding.

**Returns**

```python
{
    "base64": "...",                 # JPEG bytes, base64-encoded
    "vector": array([...], float32)  # flattened feature embedding
}
```

On error, both values come back as `None`, so callers check before continuing.

### `find_closest_match(user_vector, dataset)`

Finds the most visually similar item in the dataset by comparing feature
vectors with **cosine similarity**.

1. Stacks every non-null vector from the dataset's `Embedding` column into one
   NumPy array, so similarity is computed as a single batch operation.
2. Computes cosine similarity between `user_vector` and all dataset vectors
   using `cosine_similarity` from scikit-learn.
3. Takes the index of the highest score - the closest match.
4. Retrieves that row from the dataset.

**Returns** a tuple `(closest_row, similarity_score)`:

- `closest_row` - the DataFrame row most similar to the user's image
- `similarity_score` - a float from -1 to 1; compared against
  `SIMILARITY_THRESHOLD` to decide whether this is the *same* outfit or merely
  a *similar* one

On error, both are `None`.

---

## `models/llm_service.py`

Stage 3. This module interfaces with a large language model that understands
both text and images, configured to analyze fashion photos and write useful
responses about outfits.

**This is the one file where the watsonx-to-OpenAI swap actually happens.** The
lab reaches Llama 4 Maverick through `ibm-watsonx-ai`; here the same three
methods reach an OpenAI vision model. The method names, arguments, prompts and
returned text are unchanged - only the client underneath differs:

| Lab (watsonx) | Here (OpenAI) |
|---|---|
| `Credentials(url=region_url, api_key=...)` | `OpenAI(api_key=...)` - one client, no region |
| `APIClient(credentials)` | *(not needed)* |
| `TextChatParameters(temperature, top_p)` | per-call kwargs on `chat.completions.create` |
| `ModelInference(model_id, project_id, ...)` | `model=config.MODEL_ID` on each call |
| `model.chat(messages=...)` | `client.chat.completions.create(messages=...)` |
| `response['choices'][0]['message']['content']` | `response.choices[0].message.content` |

The API key is **required** here. watsonx made it optional inside the lab
environment; OpenAI has no such exemption, so it is read from `.env` as
`OPENAI_API_KEY`.

### `__init__`

Sets up the client and the generation settings once, then reuses them:

| Set up | From |
|---|---|
| the OpenAI client | `OPENAI_API_KEY` in `.env` |
| `model_id` | `config.MODEL_ID` |
| `temperature`, `top_p` | `config.TEMPERATURE`, `config.TOP_P` |
| `max_tokens` | `config.MAX_TOKENS` |

There is no project ID and no region, so the lab's `project_id` and `region`
arguments are gone from the signature.

### `generate_response(encoded_image, prompt)`

Generates text from the vision model given a base64 image and a prompt.

1. Builds a `messages` payload that combines the prompt as a **text part** and
   the image as an **image part**, formatted as a `data:image/jpeg;base64,...`
   URI - the shape multimodal chat endpoints expect.
2. Sends that payload to the model's chat endpoint.
3. Extracts the text from the returned object.
4. Logs both prompt length and response length, to keep model behaviour
   observable.
5. **Truncation safeguard** - if the returned content approaches typical output
   limits (around 7,900 characters), it is flagged as possibly cut off.
6. Catches and logs any exception during the call.

**Returns** the model's generated text, or a descriptive error message if the
request failed.

The multimodal message shape, which is the whole trick:

```python
"content": [
    {"type": "text",      "text": prompt},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
]
```

### `generate_fashion_response(user_image_base64, matched_row, all_items, similarity_score, threshold=0.8)`

The retrieval-augmented call: it turns the rows retrieved in stage 2 into a
prompt, and asks for a professional retail-catalog write-up.

1. Converts the `all_items` DataFrame into a **markdown list** of item names,
   prices and product links.
2. Picks one of two structured prompts, based on the similarity score:

   | Condition | Prompt asks for | Section appended |
   |---|---|---|
   | score **above** threshold | professional fashion analysis of these exact items | `ITEM DETAILS:` |
   | score **below** threshold | analysis that makes clear these are *not* exact matches | `SIMILAR ITEMS:` |

3. Sends the structured prompt plus the image through `generate_response`.
4. Applies two safety checks to whatever comes back:
   - **Too short** - a suspiciously brief reply means the model gave up or was
     cut off, so a basic response is assembled from the item descriptions
     instead.
   - **Missing its section** - if the reply lacks `ITEM DETAILS:` or
     `SIMILAR ITEMS:`, the section is appended manually.

**Returns** a markdown, catalog-style response: a formal analysis of the
clothing followed by the matched or similar item list. Those two checks mean the
retrieved item data reaches the user **even when the model misbehaves** - the
prices and links come from the dataset, not from the model, so they survive a
bad generation.

---

## `utils/helpers.py`

Three small functions that shape retrieved data into something a reader can use.
They make the application friendlier: grouping related items, and presenting
alternatives in a helpful, educational format.

### `get_all_items_for_image(image_url, dataset)`

Retrieves **all** fashion items associated with one image URL.

Filters the dataset for every row whose `Image URL` column matches, logs how
many were found, and returns those rows as a DataFrame.

Small function, load-bearing role: it is the second half of retrieval. Stage 2's
nearest-neighbour search returns a single *row*, but a row is one garment, not
an outfit. This expands that hit into the entire outfit or collection - up to 11
items sharing the same photo - which becomes the context handed to the LLM.

### `format_alternatives_response(user_response, alternatives, similarity_score, threshold=0.8)`

Enhances the model's response by appending a formatted list of similar item
alternatives, sourced from a product search rather than from the dataset.

1. Checks the original response for **refusal phrases**; if the model declined,
   that text is replaced with a generic header so the analysis still starts
   cleanly.
2. Chooses the section heading - *"Similar Items Found"* - according to whether
   the similarity score clears the threshold.
3. For each detected item, writes a subheader and lists up to **3 alternative
   products** drawn from the `alternatives` dictionary.
4. Renders each alternative with its **title, price, source and purchase link**.
5. Caps the whole section at **10 items**, to keep the response readable.
6. Where an item has no alternatives, writes a placeholder line under it rather
   than leaving a bare heading.

**Returns** a Markdown string: the model's response (or the fallback) followed by
a structured section of up to ten recommendations.

This is the "high-end fashion made more accessible" half of the app - the
dataset gives the real garment at its real price, and this section offers
comparable options at other price points.

> **Note:** `alternatives` comes from a SerpAPI-backed search service. That
> module is not part of the starter code and `SERPAPI_API_KEY` is optional in
> `.env`, so this path stays dormant until both exist. The rest of the pipeline
> runs without it.

### `process_response(response)`

Already implemented in the starter code. The final cleanup pass before display:

- Returns a placeholder if the response is empty.
- Detects refusal phrases and, rather than showing the refusal, extracts the
  `ITEM DETAILS:` / `SIMILAR ITEMS:` section so the retrieved data still reaches
  the user.
- Escapes `$` so Markdown does not read prices as LaTeX math - which is why
  `$3,350.00` survives rendering intact.
- Converts `ITEM DETAILS:` / `SIMILAR ITEMS:` into Markdown headings, adds a
  title if none is present, and normalizes `*` bullets to `-`.

---

## `app.py`

The orchestration layer: it owns the dataset, wires the components together, and
puts a Gradio interface in front of them.

### `StyleFinderApp.__init__(dataset_path, serp_api_key=None)`

Loads the dataset and builds the two components of the RAG pipeline.

| Responsibility | Detail |
|---|---|
| Load the dataset | read the pickle at `dataset_path` into `self.data` |
| Handle a missing file | raise `FileNotFoundError` before anything else runs |
| Handle an empty dataset | raise `ValueError` - an empty DataFrame would make every search silently return nothing |
| Set up the image processor | `ImageProcessor` with `IMAGE_SIZE`, `NORMALIZATION_MEAN`, `NORMALIZATION_STD` from `config.py` |
| Set up the LLM service | the vision service with `MODEL_ID`, `TEMPERATURE`, `TOP_P`, `MAX_TOKENS` |

Both components are constructed **once**, at startup, and reused for every
upload. That matters for the image processor in particular: it loads ResNet50
and its ~98 MB of weights, which should happen while the app boots rather than
on a user's first click.

The errors are raised, not swallowed, so a bad dataset path fails loudly at
launch instead of producing an app that starts fine and then fails on every
image.

### `StyleFinderApp.process_image(image)`

The full pipeline for one upload, from PIL image to finished markdown.

**Input handling** - accepts a PIL image from Gradio. If it is not already a file
path, it is saved to a temporary file, because `encode_image` reads from disk.

| Step | Call | On failure |
|---|---|---|
| 1. Encode | `encode_image` -> base64 + feature vector | vector is `None` -> return an error message |
| 2. Match | `find_closest_match` against `self.data` | no match -> return an error message |
| 3. Retrieve | `get_all_items_for_image` on the matched URL | no items -> return an error message |
| 4. Generate | `generate_fashion_response` with base64, matched row, all items, score | *(the service handles its own fallbacks)* |

**Cleanup** - any temporary file created for the upload is deleted after
processing.

**Final output** - the raw response goes through `process_response` for
post-processing, and that formatted string is returned.

**Returns** a markdown-compatible string: an LLM-generated analysis of the
uploaded photo plus the related product details. If any stage fails, a graceful
user-facing error message is returned instead of a traceback.

Note the shape of the failure handling: every step checks its result and returns
a readable message. A Gradio app has no console for the user to inspect, so an
unhandled exception would surface as an empty box.

### `create_gradio_interface(app)`

Builds the user interface:

- a **Blocks** interface with a theme and title
- an **introduction** explaining what the application does
- an **example images** section, showing the kinds of photos that work well
- **buttons to load those examples**, for quick testing without an upload
- the main **image upload** component
- a primary **analyze button**, styled as the main action
- a **status indicator**, so a multi-second model call does not look frozen
- an **output** component rendering the formatted markdown analysis
- **event handlers** connecting the buttons to `process_image`
- closing **informational sections** on how the technology works

> Built against **Gradio 6**. The lab pins `gradio==5.22.0`, and Gradio changes
> its API across major versions, so interface snippets written for v5 may not
> transfer directly.

---

## The importance of prompt engineering

Prompt engineering is the art and science of crafting effective instructions for
AI language models to produce desired outputs. As `generate_fashion_response`
demonstrates, well-designed prompts are fundamental to reliable, consistent and
useful generated content - and in a RAG pipeline they are also where the
retrieved data gets *put to work*.

### Strategic communication with AI

The fashion response prompt does four distinct jobs, each traceable to a
specific phrase:

| Job | The phrase that does it |
|---|---|
| **Define the AI's role and context** | `"You're conducting a professional retail catalog analysis"` - establishes a clear professional identity and purpose |
| **Set boundaries and focus** | `"Focus exclusively on professional fashion analysis"` - prevents the model veering into unrelated topics |
| **Structure the output** | numbered instructions plus a required `ITEM DETAILS` section - ensures consistent formatting and organization |
| **Control tone and style** | `"Use formal, clinical language"` - guides the linguistic character of the response |

There is a fifth job specific to this application, and it is the reason the
prompt is built at runtime rather than written as a constant: the retrieved
items are **interpolated into the prompt itself**. The model is not asked to
recall what a Versace blazer costs - it is handed the answer and asked to write
around it.

### Failsafe mechanisms

Prompt engineering also means not trusting the prompt. Even a well-built one can
come back short, off-format, or refused, so the function verifies what it gets:

| Check | Condition | Action |
|---|---|---|
| Minimum quality | `len(response) < 100` | discard it; build a basic response from the item data |
| Critical information present | no `ITEM DETAILS:` / `SIMILAR ITEMS:` | append the section to whatever came back |

The distinction worth holding on to: the **analysis** is generated and can fail,
but the **prices and links are retrieved**, not generated. They come from the
DataFrame. So when the model underdelivers, the fallbacks still hand the user
correct, complete item data - degraded prose, never degraded facts.

In practice the second check fires often rather than rarely. Asked for a literal
`ITEM DETAILS:` header, a model will frequently render it as its own markdown -
`#### Item Description` - which does not match the string being searched for, so
the section gets appended. The output is correct either way, which is the point
of writing the safeguard as a repair rather than an error.

---

## The dataset

`swift-style-embeddings.pkl` - a pandas DataFrame, 192 rows x 6 columns,
covering 93 distinct outfit photos.

| Column | Contents |
|---|---|
| `Image URL` | source photo URL - **the join key**; rows sharing it form one outfit |
| `Item Name` | e.g. `Versace 'Tweed Masculine Blazer'` |
| `Price` | preformatted string, `$3,350.00` (not a number) |
| `Link` | purchase link |
| `Encoded Image` | base64 of the outfit photo, precomputed |
| `Embedding` | **1000-dim float32** ResNet50 vector |

Download it into this folder:

```powershell
curl -L -o swift-style-embeddings.pkl "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/95eJ0YJVtqTZhEd7RaUlew/processed-swift-style-with-embeddings.pkl"
```

---

## Layout

```
app.py                      Gradio UI + StyleFinderApp orchestration
config.py                   model id, image size, normalization, thresholds
models/image_processor.py   stage 1 + stage 2: encoding and similarity search
models/llm_service.py       stage 3: the vision LLM call and prompt construction
utils/helpers.py            outfit lookup, alternatives formatting, response cleanup
examples/                   test-1.png ... test-6.png sample photos
```

---

## Setup

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then create `.env` in this folder (see `.env.example`):

```
OPENAI_API_KEY=sk-...
SERPAPI_API_KEY=...        # optional - only for the similar-items search
```

## Run

```powershell
python app.py
```

The Gradio interface serves on `http://127.0.0.1:5000`.

---

## Notes

**OpenAI in place of watsonx.** The original lab calls Meta's
`llama-4-maverick-17b-128e-instruct` through `ibm-watsonx-ai`. This version uses
OpenAI's vision models instead. Only `config.py` and `models/llm_service.py`
change - the pipeline, the retrieval, and the prompts are identical, and so is
the output. Any vision-capable model works here; the multimodal message shape is
the same one used in `../image_captioning`.

**Embeddings are 1000-dim, not 2048.** The stored vectors come from the *full*
unmodified ResNet50, including its final ImageNet classification layer. The
usual instinct - strip the last layer to "extract features" - yields 2048-dim
vectors and makes `cosine_similarity` fail on shape against this dataset. Run
the model as-is.

**Retrieval is visual only.** There is no text index and no vector database
here; the 192 vectors are compared in memory with scikit-learn. At this size
that is instant, and it keeps the RAG mechanics visible rather than hidden
behind a database.

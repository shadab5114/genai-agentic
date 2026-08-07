# Semantic Similarity with FAISS — Cheat Sheet

> **Topic:** Building a semantic search engine using the Universal Sentence Encoder (USE) and FAISS

---

## Table of Contents

- [What Is Semantic Search?](#what-is-semantic-search)
- [How Semantic Search Works](#how-semantic-search-works)
- [Vectors: The Language of Semantic Search](#vectors-the-language-of-semantic-search)
- [Universal Sentence Encoder (USE)](#universal-sentence-encoder-use)
- [FAISS](#faiss)
- [Environment Setup](#environment-setup)
- [Preprocessing Text Data](#preprocessing-text-data)
- [Vectorizing Documents with USE](#vectorizing-documents-with-use)
- [Indexing with FAISS](#indexing-with-faiss)
- [Querying with FAISS](#querying-with-faiss)
- [Key Takeaways](#key-takeaways)

---

## What Is Semantic Search?

Semantic search transcends the limitations of traditional keyword search by understanding the **context and nuance of language** in a query. At its core, semantic search:

- Enhances the search experience by interpreting the **intent and contextual meaning** behind a query
- Delivers more accurate and relevant results by analyzing relationships between words and phrases
- Adapts to user behavior and preferences, refining results over time

---

## How Semantic Search Works

At a conceptual level, a semantic search engine performs three steps:

1. **Getting the gist** — the engine reads the query and extracts its real meaning, rather than just spotting keywords.
2. **Making connections** — it accounts for related words and synonyms (e.g., "doctor" and "physician") to better understand intent.
3. **Picking the best matches** — it ranks all available information against the query's meaning and surfaces what's most relevant.

### The Technical Version

Under the hood, this process is powered by **vectors** — numerical representations of text meaning:

1. **Vectorization** — the query is converted into a vector as soon as it's typed.
2. **Indexing** — the engine scans a pre-built index of vectors representing other pieces of information.
3. **Retrieval** — the engine finds the closest matching vectors and returns them as results — content that is semantically related, not just textually similar.

---

## Vectors: The Language of Semantic Search

A **vector** is a list of numbers a computer uses to represent the meaning of a word or sentence. Each piece of text becomes a point in a high-dimensional space; the closer two points are, the more similar their meanings.

- **Creating vectors** — models like the Universal Sentence Encoder turn text into a unique numerical "fingerprint."
- **Calculating similarity** — closeness between vectors is measured with distance/similarity metrics such as **cosine similarity** or **Euclidean (L2) distance**.
- **Using vectors for search** — a search engine finds the vectors closest to the query's vector; those represent the most relevant results.

---

## Universal Sentence Encoder (USE)

USE takes sentences — however complex — and turns them into fixed-length vectors that capture their semantic essence.

| Property | Description |
|---|---|
| **Language comprehension** | Understands sentence meaning by considering the context each word is used in |
| **Versatility** | Trained on diverse data sources, so it handles a wide range of topics and sentence structures |
| **Speed** | Once trained, converts sentences to vectors quickly and efficiently |

**How it works internally:**

1. **Analyzes words** — looks at each word and its surrounding words to build a full picture of meaning.
2. **Understands context** — attends to word order and usage to grasp sentence intent.
3. **Creates vectors** — encodes all of this understanding into a single numeric vector per sentence.

---

## FAISS

**FAISS** (Facebook AI Similarity Search) is a library for efficient similarity search over dense vectors.

| Property | Description |
|---|---|
| **Efficient searching** | Uses optimized algorithms to rapidly search large collections of vectors |
| **Scalability** | Can handle vector databases too large to fit in memory |
| **Accuracy** | Provides highly accurate results via advanced indexing strategies |

**How it works:**

1. **Index building** — organizes vectors so similar ones are stored near each other, speeding up matching.
2. **Searching** — given a new query vector, quickly identifies which part of the index to examine for the closest matches.
3. **Retrieving results** — returns the most similar vectors, corresponding to the most relevant search results.

Together, USE converts language into numbers, and FAISS searches through those numbers efficiently — combining them produces a semantic search engine that is both smart and fast.

---

## Environment Setup

Required libraries:

| Library | Purpose |
|---|---|
| `tensorflow` | Core library required for working with the Universal Sentence Encoder |
| `tensorflow-hub` | Downloads and deploys pre-trained TensorFlow models, including USE |
| `faiss-cpu` | Efficient similarity search and clustering of dense vectors |
| `numpy` | Numerical computing — handling arrays and matrices |
| `scikit-learn` | Data mining/analysis tools — used here for the sample dataset and preprocessing helpers |

```bash
pip install "tensorflow>=2.0.0"
pip install faiss-cpu numpy scikit-learn
pip install --upgrade tensorflow-hub
```

```python
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import faiss
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

# Suppress warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
```

---

## Preprocessing Text Data

Preprocessing improves data quality before text is fed into the embedding model. A typical pipeline:

1. **Fetch data** — load raw documents into a list.
2. **Remove email headers** — strip metadata lines (e.g., `From:` headers).
3. **Remove email addresses** — strip patterns resembling email addresses.
4. **Strip punctuation and numbers** — keep only alphabetic characters, reducing noise.
5. **Lowercase** — standardize casing for uniformity.
6. **Trim excess whitespace** — collapse extra spaces, tabs, and line breaks.

```python
def preprocess_text(text):
    # Remove email headers
    text = re.sub(r'^From:.*\n?', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

documents = [...]  # raw text documents
processed_documents = [preprocess_text(doc) for doc in documents]
```

> **Note:** Cleaning and standardizing text reduces noise so the embedding model can focus on meaningful content, improving the quality of the resulting vectors.

---

## Vectorizing Documents with USE

Once text is cleaned, it's converted into vectors using the pre-trained Universal Sentence Encoder.

```python
# Load the Universal Sentence Encoder's TF Hub module
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Function to generate embeddings
def embed_text(text):
    return embed(text).numpy()

# Generate embeddings for each preprocessed document
X_use = np.vstack([embed_text([doc]) for doc in processed_documents])
```

- `embed(text)` converts text into a high-dimensional vector capturing its semantic meaning.
- `.numpy()` converts the TensorFlow tensor into a NumPy array for downstream use.
- `np.vstack([...])` stacks all document vectors into a single 2D array (`X_use`), where each row is one document's embedding — ready for FAISS indexing.

---

## Indexing with FAISS

```python
dimension = X_use.shape[1]
index = faiss.IndexFlatL2(dimension)  # Index using L2 (Euclidean) distance
index.add(X_use)                      # Add document vectors to the index
```

- `IndexFlatL2` performs an exact (brute-force) search using Euclidean distance — simple and effective for small to medium-sized datasets.
- FAISS provides many other index types for different scales and trade-offs between speed, memory, and accuracy. For larger datasets or more advanced needs, consider:

| Index Type | Use Case |
|---|---|
| `IndexFlatL2` | Exact search, small–medium datasets |
| `IndexIVFFlat` | Faster approximate search via inverted-file clustering, larger datasets |
| `IndexIVFPQ` | Reduced memory usage via product quantization, very large datasets |

---

## Querying with FAISS

To search, a query goes through the same preprocessing and embedding pipeline as the documents, then FAISS finds its nearest neighbors in the index.

```python
def search(query, k=5):
    # Preprocess the query the same way documents were processed
    processed_query = preprocess_text(query)

    # Convert query to a vector
    query_vector = embed_text([processed_query])

    # Search the FAISS index for the k nearest neighbors
    distances, indices = index.search(query_vector, k)

    return distances, indices

# Example query
distances, indices = search("motorcycle")

for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
    print(f"Rank {rank} | Distance: {dist:.4f}")
    print(f"Preprocessed: {processed_documents[idx][:200]}...")
    print(f"Original: {documents[idx][:200]}...")
    print("-" * 80)
```

- **`distances`** — how close each match is to the query vector (lower L2 distance = more similar).
- **`indices`** — positions of the matching documents in the original dataset.
- Results are ranked by distance, so rank 1 is the most semantically similar document to the query.

This demonstrates the core value of semantic search: results are retrieved based on **meaning**, not just literal keyword overlap — a query like `"motorcycle"` can surface relevant documents even if they use different but related terminology.

---

## Key Takeaways

- ✅ Semantic search interprets the **meaning and intent** behind a query, not just its literal keywords.
- ✅ Text is converted into **vectors** (numerical fingerprints of meaning); similarity between vectors approximates similarity in meaning.
- ✅ The **Universal Sentence Encoder** produces high-dimensional sentence embeddings that capture context, not just individual words.
- ✅ **FAISS** provides fast, scalable, and accurate nearest-neighbor search over large collections of vectors.
- ✅ Preprocessing (removing noise, standardizing case/whitespace) is essential before embedding — messy input produces messy vectors.
- ✅ `IndexFlatL2` is a good default for small–medium datasets; larger-scale applications should consider approximate indexes like `IndexIVFFlat` or `IndexIVFPQ` for better speed/memory trade-offs.
- ✅ Query text must go through the **same preprocessing and embedding pipeline** as the indexed documents to produce comparable vectors.

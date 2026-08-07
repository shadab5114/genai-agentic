# Vector Databases for Recommendation Systems and RAG — Cheat Sheet

> **Estimated reading time:** 15 minutes
> **Source:** cognitiveclass.ai — Wojciech "Victor" Fulmyk

---

## Table of Contents

- [What is RAG?](#what-is-rag)
- [RAG Pipeline Steps](#rag-pipeline-steps)
- [Vector Database Responsibilities in RAG](#vector-database-responsibilities-in-rag)
- [Why Use Vector Databases for RAG Steps](#why-use-vector-databases-for-rag-steps)
- [Common RAG Pipeline Pitfalls](#common-rag-pipeline-pitfalls)
- [Chroma DB Operations](#chroma-db-operations)
- [Distance Functions in Chroma DB](#distance-functions-in-chroma-db)
- [What Vector Databases Don't Handle](#what-vector-databases-dont-handle)
- [RAG Frameworks](#rag-frameworks)
- [Key Takeaways](#key-takeaways)

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a framework that enhances language models by retrieving relevant information from external sources and using it to generate more accurate, grounded responses — reducing the amount of potential hallucination.

### Core Problems RAG Solves

| Problem | Description |
|---|---|
| **Limited context windows** | It's not possible to include all relevant information in a single prompt. |
| **Frozen knowledge** | LLM knowledge is fixed at the point it was trained. |
| **Hallucination** | LLMs can generate facts that are incorrect or fabricated. |

---

## RAG Pipeline Steps

1. Provide source documents relevant to the use case, and split them into smaller chunks if needed.
2. Embed the source documents or their chunks.
3. Store the sources and their embeddings in a vector database (e.g., Chroma DB).
4. Receive the user's prompt.
5. Embed the user's prompt.
6. Use a retriever to select the chunks from the vector store that best match the prompt.
7. Combine the retrieved text with the original prompt to produce an **augmented prompt**.
8. Pass the augmented prompt to the LLM to produce a context-aware response.

---

## Vector Database Responsibilities in RAG

A vector database can handle several key responsibilities:

- Embedding both source documents and user prompts
- Storing those embeddings
- Retrieving the most relevant matches
- Supplying the retrieved content for prompt augmentation

> **Note:** Steps 2 and 5 (embedding documents and prompts) can also be performed externally. In that case, the vector database is used primarily for **storing and retrieving** vectors.

---

## Why Use Vector Databases for RAG Steps

| Benefit | Explanation |
|---|---|
| **Prevents critical mistakes** | Avoids accidentally using different embedding models for source documents vs. user prompts, or mislinking embeddings to their source documents. |
| **Faster, cleaner development** | Fewer moving parts and less custom logic keeps the codebase simpler, easier to maintain, and faster to debug. |
| **Superior performance** | Built for high-speed, scalable semantic search using advanced indexing algorithms — hard to match with custom-built alternatives without significant optimization effort. |

---

## Common RAG Pipeline Pitfalls

⚠️ **Critical mistakes to avoid:**

1. **Mismatched embedding models**
   Using different embedding models for documents and queries can break retrieval entirely.
   → *Solution:* Use the same embedding model throughout. Vector databases usually handle this automatically.

2. **Poor chunking strategy**
   Chunks that are too large or too small hurt retrieval quality.
   → *Solution:* Choose a chunk size long enough to preserve meaning, without including too much irrelevant content.

3. **Forgetting to re-embed after config changes**
   Changing the distance metric or embedding model requires re-embedding existing content.
   → *Note:* In Chroma DB, this can't be done on an existing collection — you may need to clone the collection.

4. **Assuming the top retrieved result is always correct**
   → *Solution:* Always test your results — a little tuning can make a big difference.

---

## Chroma DB Operations

### Creating Collections

```python
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

# Define the embedding model
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Define chromadb client
client = chromadb.Client()

# Create collection
collection = client.create_collection(
    name="my_collection",
    metadata={"description": "A collection for storing user data"},
    configuration={
        "embedding_function": sentence_transformer_ef
    }
)
```

### Connecting to Existing Collections

```python
collection = client.get_collection(name="my_collection")
```

### Modifying Collections

```python
collection.modify(
    name="new_collection_name",
    metadata={"key": "value"}
)
```

> ⚠️ **Important:** Changes to the **embedding model** or **distance metric** cannot be applied to an existing collection. You must clone the collection instead — this can be computationally expensive for large datasets.

### Adding Documents

```python
collection.add(
    documents=[
        "This is a document about LangChain",
        "This is a document about LlamaIndex"
    ],
    metadatas=[
        {"source": "langchain.com", "version": "0.2"},
        {"source": "llamaindex.ai", "version": "0.12"}
    ],
    ids=["id1", "id2"]
)
```

> **Note:** Always include an `id` for each document in the `ids` parameter.

### Retrieving Documents

```python
# Get all documents (returns a Python dictionary)
results = collection.get()

# Get specific documents by ID
results = collection.get(ids=["id1"])

# Include embeddings in results
results = collection.get(include=['embeddings'])
```

> **Note:** `get()` does not return embeddings by default (to keep output clean), but they are still stored in the collection.

### Updating Documents

```python
collection.update(
    ids=["id1"],
    metadatas=[{"source": "langchain.com", "version": "0.3"}],
    documents=["This an updated document about LangChain"]
)
```

> ⚠️ **Important:** Chroma DB automatically re-embeds documents in the background as soon as an update is submitted.

### Deleting Documents

```python
# Delete by IDs
collection.delete(ids=["id1"])

# Delete using metadata filter
collection.delete(where={"source": "doc_to_delete.pdf"})

# Combine IDs and filters
collection.delete(
    ids=["id1"],
    where={"version": "1.0"}
)
```

---

## Distance Functions in Chroma DB

Chroma DB uses the **Hierarchical Navigable Small World (HNSW)** algorithm for approximate nearest neighbor search.

The `space` parameter defines the distance function used in the embedding space:

| Value | Distance Function | Notes |
|---|---|---|
| `l2` | Squared L2 norm | **Default** |
| `cosine` | Cosine distance | Common for semantic similarity |
| `ip` | Inner product / dot product distance | |

### Configuring the Distance Function

```python
collection = client.create_collection(
    name="my_collection",
    metadata={"description": "A collection for storing user data"},
    configuration={
        "embedding_function": sentence_transformer_ef,
        "hnsw": {"space": "cosine"}
    }
)
```

---

## What Vector Databases Don't Handle

Some RAG pipeline tasks usually happen **outside** the vector database:

- 🔹 **Chunking** — typically done before data enters the vector database
- 🔹 **Advanced retrieval logic** — filtering and re-ranking may require extra tools
- 🔹 **Prompt augmentation** — typically handled outside the database
- 🔹 **LLM integration** — not built into most vector databases

---

## RAG Frameworks

Tools like **LangChain** and **LlamaIndex** wrap around your vector database and help manage the pipeline from document preparation to final response. These frameworks:

- Provide additional structure
- Simplify development and deployment of RAG applications
- Fill the gaps by connecting all the pieces

---

## Key Takeaways

- ✅ Vector databases are the foundation that makes retrieval-augmented generation work.
- ✅ RAG enhances LLM response quality by retrieving relevant external information, helping the model generate more accurate and well-supported outputs.
- ✅ Using a vector database for all relevant RAG steps helps prevent critical mistakes, speeds up development, and optimizes performance.
- ✅ **Collections** are how you organize data within Chroma DB.
- ✅ **Metadata** helps track the purpose and contents of a collection.
- ✅ Always test your results — a little tuning can make a big difference.

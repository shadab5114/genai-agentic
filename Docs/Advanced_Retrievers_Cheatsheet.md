# Advanced Retrievers for RAG — Cheat Sheet

> **Estimated reading time:** 15 minutes
> **Source:** cognitiveclass.ai — Wojciech "Victor" Fulmyk

---

## Table of Contents

- [What Are Advanced Retrievers?](#what-are-advanced-retrievers)
- [Maximum Marginal Relevance (MMR)](#maximum-marginal-relevance-mmr)
- [LlamaIndex: Core Index Types](#llamaindex-core-index-types)
- [LlamaIndex Retriever Types](#llamaindex-retriever-types)
- [LangChain Retriever Interface](#langchain-retriever-interface)
- [LangChain Retriever Types](#langchain-retriever-types)
- [Decision Framework](#decision-framework)

---

## What Are Advanced Retrievers?

Advanced retrievers go beyond simple vector similarity search to provide more nuanced, context-aware information retrieval through:

- **Semantic Understanding** — using embeddings for meaning and context
- **Keyword Matching** — precise term-based search for exact specifications
- **Hierarchical Context** — maintaining relationships between information levels
- **Multi-Query Processing** — generating and combining results from multiple query variations
- **Fusion Techniques** — intelligently combining results from different retrieval methods

---

## Maximum Marginal Relevance (MMR)

| Aspect | Detail |
|---|---|
| **Purpose** | Balance relevance and diversity of retrieved results |
| **Method** | Selects documents that are highly relevant to the query **and** minimally similar to previously selected documents |
| **Benefit** | Avoids redundancy and ensures comprehensive coverage of different aspects of the query |

---

## LlamaIndex: Core Index Types

### VectorStoreIndex

- **Function:** Stores vector embeddings for each document chunk
- **Best suited for:** Semantic retrieval based on meaning
- **Usage:** Commonly used in LLM pipelines and RAG applications

### DocumentSummaryIndex

- **Function:** Generates and stores summaries of documents at indexing time
- **Process:** Uses summaries to find and retrieve relevant documents
- **Best for:** Large documents whose meaning would be lost by chunking, or documents too large to fit in an LLM's or embedding model's context window
- **Key points:** Returns the *original documents*, not their summaries — summaries are used instead of text chunks purely to enable retrieval based on the semantic meaning of the entire text

### KeywordTableIndex

- **Function:** Extracts keywords from documents and maps them to content chunks
- **Best for:** Exact keyword matching for rule-based or hybrid search scenarios
- **Use case:** Applications requiring precise term matching

---

## LlamaIndex Retriever Types

### 1. Vector Index Retriever

The most common retriever — uses vector embeddings to find semantically related content.

- **Process:** Embeds the query, compares it with document embeddings using cosine similarity
- **Ideal for:** General-purpose search, RAG pipelines where semantic understanding is crucial
- **Limitation:** May miss exact keyword matches when specific terms are crucial

### 2. BM25 Retriever

Advanced keyword-based retrieval that improves on TF-IDF.

**TF-IDF foundation:**

- **Term Frequency (TF):** How often a word appears in a document
- **Inverse Document Frequency (IDF):** How rare a word is across all documents
- **TF-IDF score:** `TF × IDF`

**BM25 improvements over TF-IDF:**

- **Term Frequency Saturation:** Reduces the impact of repeated terms using a saturation function
- **Document Length Normalization:** Adjusts for document length, preventing bias toward longer documents
- **Tunable Parameters:** `k1 ≈ 1.2` (saturation control), `b ≈ 0.75` (length normalization)

**Best for:** technical documentation, legal documents, exact terminology requirements

### 3. Document Summary Index Retrievers

Two variants:

- **`DocumentSummaryIndexLLMRetriever`** — uses an LLM to analyze the query against summaries (intelligent but expensive)
- **`DocumentSummaryIndexEmbeddingRetriever`** — uses semantic similarity between query and summary embeddings (faster, cost-effective)

**Process:** Two-stage approach — uses summaries to filter down to relevant documents, then returns the full document content.

### 4. Auto Merging Retriever

- **Purpose:** Preserves context in long documents using hierarchical structure
- **Method:**
  - Uses hierarchical chunking (parent and child nodes)
  - If enough child nodes from the same parent are retrieved, returns the parent node instead
- **Dual storage:** Child chunks for precise matching, parent chunks for context
- **Best for:** Long documents, legal papers, technical specifications needing context preservation

### 5. Recursive Retriever

- **Purpose:** Follows relationships between nodes using references
- **Capability:** Can follow references from one node to another (citations, metadata links)
- **Types:** Supports chunk references and metadata references
- **Best for:** Academic papers with citations, interconnected knowledge bases

### 6. Query Fusion Retriever

Combines results from different retrievers and optionally generates multiple query variations.

**Core capabilities:**

- Multiple retriever support (combines vector-based and keyword-based methods)
- Query variation generation using an LLM
- Sophisticated fusion strategies to improve recall

**Three fusion modes:**

**Reciprocal Rank Fusion (RRF)**
- Most robust fusion method — combines ranked lists using the reciprocal of ranks
- Formula: `RRF_score(d) = Σ (1 / (rank_i(d) + k))`, where `k ≈ 60`
- Best for: default choice for most fusion scenarios, production systems

**Relative Score Fusion**
- Preserves score magnitudes while normalizing across query variations
- Formula: `normalized_score = original_score / max_score`
- Best for: when embedding model confidence scores are meaningful

**Distribution-Based Score Fusion**
- Most sophisticated — uses statistical properties of score distributions
- Methods: Z-score normalization, percentile ranking
- Best for: complex queries with varying score distributions

---

## LangChain Retriever Interface

> **Definition:** "An interface that returns documents based on an unstructured query."

- More general than a vector store
- Accepts a string query as input, returns a list of documents as output
- Doesn't necessarily store documents — its purpose is to *retrieve* them

---

## LangChain Retriever Types

### 1. Vector Store-Backed Retriever

The foundation retriever — a lightweight wrapper around a vector store class.

**Search types:**

- **Simple Similarity Search:** Returns documents ranked by similarity (default 4 results)
- **MMR Search:** Balances relevance and diversity to avoid redundancy
- **Similarity Score Threshold:** Returns only documents above a specified threshold

### 2. Multi-Query Retriever

**Problem addressed:** "Distance-based vector database retrieval may vary with subtle changes in query wording."

**Solution process:**

1. Uses an LLM to generate multiple queries from different perspectives
2. For each query, retrieves a set of relevant documents
3. Takes the unique union of results for a larger set of potentially relevant documents

**Benefit:** "By generating multiple perspectives on the same question, the MultiQueryRetriever can potentially overcome some limitations of distance-based retrieval."

### 3. Self-Querying Retriever

**Core capability:** "Has the ability to query itself."

**Process:** Converts a natural language query into a structured query with two components:

1. A string to look up semantically
2. A metadata filter to accompany it

**Requirements:** Documents must have rich, structured metadata with field descriptions.

**Best for:** applications combining semantic search with attribute filtering.

**Example queries:**

- *"I want to watch a movie rated higher than 8.5"* (filter only)
- *"Has Greta Gerwig directed any movies about women"* (query + filter)

### 4. Parent Document Retriever

**Problem solved:** "Conflicting desires" when splitting documents:

- Small documents for accurate embeddings
- Large documents for context retention

**Solution:** "Strikes that balance by splitting and storing small chunks of data."

**Process:**

1. During retrieval, first fetches small chunks
2. Looks up parent IDs for those chunks
3. Returns larger documents containing the small chunks

**Architecture:**

- **Two splitters:** Parent (large chunks for retrieval) and child (small chunks for embeddings)
- **Dual storage:** Vector store for embeddings, document store for parent documents

---

## Decision Framework

| Need | LlamaIndex Choice | LangChain Choice |
|---|---|---|
| **Exact keyword matching** | BM25 Retriever | Vector Store-Backed + custom keyword logic |
| **Multi-query with fusion** | Query Fusion Retriever (RRF / Relative / Distribution) | Multi-Query Retriever (union approach) |
| **Citation following** | Recursive Retriever | Not directly supported |
| **Hierarchical context** | Auto Merging Retriever | Parent Document Retriever |
| **Simple semantic search** | Vector Index Retriever | Vector Store-Backed Retriever |

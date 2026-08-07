# FAISS vs. Chroma DB — Cheat Sheet

> **Topic:** Comparing vector search technologies for RAG applications — FAISS, Chroma DB, and Milvus

---

## Table of Contents

- [FAISS (Facebook AI Similarity Search)](#faiss-facebook-ai-similarity-search)
- [Chroma DB](#chroma-db)
- [Technology Comparison Summary](#technology-comparison-summary)
- [FAISS Index Types](#faiss-index-types)
- [HNSW Deep Dive](#hnsw-deep-dive)
- [Extending FAISS with Milvus](#extending-faiss-with-milvus)
- [When to Use Each Technology](#when-to-use-each-technology)
- [Key Concepts Summary](#key-concepts-summary)

---

## FAISS (Facebook AI Similarity Search)

FAISS is a library developed by Meta for fast vector search that runs on a single machine using either CPU or GPU.

| Property | Detail |
|---|---|
| **Type** | Library for vector search operations |
| **Deployment** | Single-node operation, no native distributed scaling |
| **Usage** | Code-based integration, no server component |
| **Control** | Full control over indexing and performance |
| **Metadata** | No native metadata support |
| **Integration** | Works with LangChain and LlamaIndex |

---

## Chroma DB

Chroma DB is a vector database built for AI use cases that stores both vectors and metadata (tags, descriptions, etc). It can run locally or as a server.

| Property | Detail |
|---|---|
| **Type** | Full vector database |
| **Deployment** | Supports both single-node and distributed deployments |
| **Scaling** | Clear path to scale for larger workloads |
| **Metadata** | Native support for storing and filtering metadata |
| **Indexing** | Only supports HNSW (Hierarchical Navigable Small World) |
| **Integration** | Works well with tools like LangChain, easy to integrate |

---

## Technology Comparison Summary

| Dimension | FAISS | Chroma DB |
|---|---|---|
| **Type** | Library | Full database |
| **Deployment** | Single-node only | Single-node and distributed |
| **Indexing options** | Many (Flat, IVF, LSH, HNSW, ...) | HNSW only |
| **Metadata support** | None native | Native support and filtering |
| **Framework integration** | Works with LangChain and LlamaIndex | Works with LangChain and LlamaIndex |

---

## FAISS Index Types

### Flat Index

Compares the distance (Euclidean distance or dot product) between the query embedding and **every** vector in the store using brute-force search.

| Property | Detail |
|---|---|
| **Method** | Brute-force comparison against all vectors |
| **Accuracy** | Very accurate |
| **Performance** | Very slow for large datasets |
| **Use case** | Small datasets where accuracy is critical |

### Inverted File Index (IVF)

Speeds up search by clustering vectors (e.g., via k-means) into **Voronoi cells** around centroids; each cell holds the vectors closest to its centroid.

| Property | Detail |
|---|---|
| **Clustering** | Vectors grouped via k-means into Voronoi cells |
| **Search strategy** | Query only searches the nearest cell(s), reducing computation |
| **Trade-off** | Faster than a flat index but may slightly reduce accuracy |
| **Limitation** | Some nearby vectors may end up in a different cell than the query |

### Locality-Sensitive Hashing (LSH)

Uses hash functions to map similar vectors into the same bucket, enabling fast, memory-efficient search.

| Property | Detail |
|---|---|
| **Method** | Hash functions group similar vectors into buckets |
| **Performance** | Fast and memory-efficient |
| **Best use** | High-dimensional sparse data, such as text embeddings |
| **Trade-off** | Neither the fastest nor the most accurate method |
| **Search process** | Searches only vectors in the closest matching bucket(s) |

### Hierarchical Navigable Small World (HNSW)

Organizes vectors into a hierarchy of layers, where top layers are sparse and act like express highways, helping the search algorithm quickly approach the target region.

| Property | Detail |
|---|---|
| **Top layers** | Sparse, contain only a few vectors ("express highways") |
| **Lower layers** | Denser graphs with detailed local connections |
| **Search process** | Begins at the topmost layer, descends using the best candidate from each layer |
| **Performance** | Both fast and accurate, especially for large datasets |

> For the full mechanics, diagrams, and worked example, see [HNSW_Cheatsheet.md](HNSW_Cheatsheet.md).

---

## HNSW Deep Dive

### How HNSW Works — Technical Process

HNSW constructs a multi-layer graph where each layer down contains progressively more data points with shorter-range connections:

1. **Top layer entry** — search begins at the highest layer, which has sparse, long-range connections.
2. **Greedy search** — at each layer, move to the neighbor closest to the target query.
3. **Layer descent** — when no closer neighbor exists in the current layer, descend to the next.
4. **Progressive refinement** — the graph becomes denser as the search descends.
5. **Bottom layer** — Layer 0 contains all data points, enabling a final, precise search.
6. **Result** — returns approximate nearest neighbors with high accuracy.

> **Navigation analogy:** Like finding a restaurant in a city — start with highways for long jumps, then switch to local streets for precise location. The algorithm "zooms in" progressively from major connections to fine-grained local ones.

### Key Parameters

| Parameter | Purpose | Trade-off |
|---|---|---|
| **M** (max connections) | Controls how many neighbors each point connects to | Higher `M` = better accuracy, more memory. Lower `M` = faster build, less memory, lower accuracy |
| **efConstruction** (search breadth during build) | Controls how many candidates are considered when finding neighbors during insertion | Higher = better graph quality, slower build. Lower = faster build, possibly lower later search quality |
| **efSearch** (search breadth during querying) | Controls how many candidate nodes are explored during a query | Higher = better accuracy, slower search. **Main knob for tuning speed vs. accuracy at query time.** |
| **ml** (level multiplier) | Affects how likely a point is to appear in higher layers | Controls the overall shape of the hierarchy |

### HNSW Limitations

**Approximate results**
- Typical recall rates of 90% to 99%.
- May occasionally miss the exact nearest neighbor.
- For most applications, this trade-off is worthwhile.

**Dynamic updates**
- Best suited for mostly-static datasets.
- Frequent insertions/deletions can degrade index performance over time.
- Periodic reconstruction may be needed for optimal performance.

**Distance metric limitations**
- Works best with Euclidean distance (L2) and cosine similarity.
- Other metrics may require modification or perform suboptimally.

---

## Extending FAISS with Milvus

### FAISS Limitations

FAISS is effective for local, high-performance vector search, but it lacks features like metadata support and distributed scaling.

### Milvus Integration

**Milvus**, a vector database, uses FAISS as one of its core indexing engines while adding the capabilities FAISS lacks:

| Capability | Description |
|---|---|
| **Metadata support** | Storing and filtering metadata alongside vectors |
| **Hybrid queries** | Enables queries such as "find similar items under $50" |
| **Distributed deployments** | Suitable for large-scale production environments |
| **Scalability** | Addresses FAISS's single-node limitation |

---

## When to Use Each Technology

### Use FAISS when:
- You want full control and performance on a single machine
- You need access to multiple indexing algorithms
- You're building custom, high-performance applications
- Metadata support is not required, or can be handled externally

### Use Chroma DB when:
- You need quick AI development and prototyping
- Metadata-rich queries are important
- You want easy integration with AI tools
- You need both single-node and distributed deployment options

### Use Milvus when:
- You need a scalable, production-ready vector database
- Hybrid search capabilities are desired
- Distributed capabilities are essential
- You want FAISS-level performance combined with database features

---

## Key Concepts Summary

### Index Selection Strategy

Each FAISS index type balances speed, memory, and accuracy differently:

| Index | Characteristic | Best For |
|---|---|---|
| **Flat** | Most accurate, slowest | Small datasets only |
| **IVF** | Balanced speed/accuracy | Medium–large datasets, though HNSW often outperforms it |
| **LSH** | Memory-efficient | High-dimensional sparse data, though less commonly used |
| **HNSW** | Fast and accurate | Medium–large datasets, due to strong performance and scalability |

### HNSW Algorithm Benefits

- **Hierarchical structure** — multi-layer approach enables `O(log n)` search complexity
- **Performance** — 90–99% recall rates with fast search times
- **Scalability** — especially effective for large datasets
- **Flexibility** — tunable parameters for different speed/accuracy requirements

### Technology Integration

- Both **FAISS** and **Chroma DB** work with **LangChain** and **LlamaIndex** for RAG pipelines.
- Choice should be based on project size, complexity, and infrastructure requirements.
- **Milvus** uses FAISS under the hood while providing additional database-level capabilities — a natural extension point when FAISS's single-node/no-metadata limitations become blockers.

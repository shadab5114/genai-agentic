# Advanced Retrievers in LlamaIndex — Cheat Sheet

> **Topic:** Core and advanced retrieval techniques for building production-grade RAG applications with LlamaIndex

---

## Table of Contents

- [Background](#background)
- [What Are Advanced Retrievers?](#what-are-advanced-retrievers)
- [Why Advanced Retrievers Matter](#why-advanced-retrievers-matter)
- [Index Types Overview](#index-types-overview)
- [Vector Index Retriever](#vector-index-retriever)
- [BM25 Retriever](#bm25-retriever)
- [Document Summary Index Retriever](#document-summary-index-retriever)
- [Auto Merging Retriever](#auto-merging-retriever)
- [Recursive Retriever](#recursive-retriever)
- [Query Fusion Retriever](#query-fusion-retriever)
- [Fusion Mode Comparison](#fusion-mode-comparison)
- [Recommended Retrievers by Use Case](#recommended-retrievers-by-use-case)
- [Key Takeaways](#key-takeaways)

---

## Background

Modern RAG applications need to handle complex queries, combine multiple search strategies, and deliver precise results at scale. A single retrieval strategy — plain vector similarity search — is often not enough: it can miss exact keyword matches, lose context from long documents, fail to follow cross-references, or under-serve queries that are better answered by combining several signals.

LlamaIndex provides a set of **core retrievers** (each backed by a specific index type) plus **fusion techniques** for combining multiple retrievers or query variations into a single, higher-quality result set.

---

## What Are Advanced Retrievers?

Advanced retrievers go beyond basic top-k vector similarity search by adding one or more of:

- **Alternative relevance signals** (e.g., keyword/BM25 scoring instead of, or alongside, embeddings)
- **Hierarchical context handling** (retrieving small chunks but returning larger, more complete context)
- **Reference following** (traversing citations/links between documents)
- **Pre-filtering** (narrowing to a relevant document subset before fine-grained search)
- **Fusion** (combining multiple retrievers or query rewordings into one ranked result)

---

## Why Advanced Retrievers Matter

| Benefit | Explanation |
|---|---|
| **Precision** | Combining semantic and keyword signals reduces both missed matches and irrelevant results. |
| **Context preservation** | Hierarchical retrievers avoid returning fragments stripped of surrounding context. |
| **Scalability** | Pre-filtering strategies (e.g., document summary retrieval) keep fine-grained search fast over large collections. |
| **Robustness** | Fusion techniques reduce sensitivity to any single query's exact phrasing or any single retriever's blind spots. |

---

## Index Types Overview

Each retriever in LlamaIndex is backed by a specific index structure:

| Index Type | Backing Retriever(s) |
|---|---|
| **VectorStoreIndex** | Vector Index Retriever |
| **Keyword/BM25 index** | BM25 Retriever |
| **DocumentSummaryIndex** | Document Summary Index Retriever |
| **Hierarchical node structure** | Auto Merging Retriever |
| **Node relationships / references** | Recursive Retriever |
| **Any combination of the above** | Query Fusion Retriever |

---

## Vector Index Retriever

The foundational retriever — embeds documents/chunks and the query into the same vector space, then returns the top-k nearest neighbors by embedding similarity.

**Best for:** general semantic understanding — finding text that means the same thing as the query, even with different wording.

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=3)

nodes = retriever.retrieve("What is the smoking policy?")
```

---

## BM25 Retriever

A classic sparse, keyword-based ranking function (Best Matching 25). Unlike embeddings, BM25 scores documents based on term frequency and inverse document frequency — it excels at matching exact terms, codes, names, and jargon that embeddings can blur together.

**Best for:** technical documentation and any domain where exact terminology matters more than paraphrase-level similarity.

```python
from llama_index.retrievers.bm25 import BM25Retriever

bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=3
)

results = bm25_retriever.retrieve("error code E4021")
```

---

## Document Summary Index Retriever

Builds a summary for each document, then retrieves at the **document level** first (using the summaries) before drilling into the matching documents. This two-stage approach narrows a large collection down to a relevant subset before doing detailed retrieval.

**Best for:** large document collections, where searching every chunk of every document directly would be slow or noisy.

```python
from llama_index.core import DocumentSummaryIndex

doc_summary_index = DocumentSummaryIndex.from_documents(documents, llm=llm)
retriever = doc_summary_index.as_retriever(similarity_top_k=2)

nodes = retriever.retrieve("What do our policy documents say about remote work?")
```

---

## Auto Merging Retriever

Works over a **hierarchical node structure**: documents are split into small child chunks, which are grouped under larger parent chunks. During retrieval, if enough child chunks from the same parent are retrieved (above a merge threshold), the retriever automatically "merges" them and returns the larger parent chunk instead of several disjointed fragments.

**Best for:** long documents, where returning the full parent context (instead of scattered small chunks) produces more coherent, complete answers.

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import VectorStoreIndex, StorageContext

node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
storage_context = StorageContext.from_defaults(docstore=docstore)

base_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
base_retriever = base_index.as_retriever(similarity_top_k=6)

retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)
results = retriever.retrieve("Summarize the onboarding process")
```

---

## Recursive Retriever

Follows **references between nodes** — links, citations, or explicit node relationships — recursively pulling in connected content rather than treating every chunk as an isolated island. Starting from an initial retrieval, it can traverse to referenced nodes (e.g., a citation, a linked sub-document, or an index-node pointing to another retriever) and fold their content into the result.

**Best for:** research papers and any corpus with citation graphs or cross-references, where the answer may depend on content the top match merely points to.

```python
from llama_index.core.retrievers import RecursiveRetriever

recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": base_retriever},
    node_dict=node_mappings,
    verbose=True
)

results = recursive_retriever.retrieve("What method did the cited paper use?")
```

---

## Query Fusion Retriever

Combines results from **multiple retrievers** and/or **multiple query variations** into a single ranked list. It can generate reworded versions of the input query (similar to multi-query expansion), run each retriever against each query variation, and then fuse all the resulting ranked lists into one combined ranking.

**Best for:** hybrid search (e.g., Vector + BM25) and reducing sensitivity to a single query's exact phrasing.

```python
from llama_index.core.retrievers import QueryFusionRetriever

fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=4,          # original + 3 generated variations
    mode="reciprocal_rerank",
    use_async=True,
    verbose=True,
)

results = fusion_retriever.retrieve("What is our remote work policy?")
```

### Fusion Modes

Query Fusion supports several strategies for combining scores across retrievers/query variations:

**1. Reciprocal Rank Fusion (RRF)** — `mode="reciprocal_rerank"`
Combines results based on their **rank position** in each list, not raw scores. This makes it scale-invariant — it doesn't matter that one retriever's scores range 0–1 and another's range 0–100.

**2. Relative Score Fusion** — `mode="relative_score"`
Normalizes each result's score by the **maximum score** in its query's result set, preserving relative confidence within each query before combining.

**3. Distribution-Based Score Fusion** — the most statistically sophisticated approach:

```python
import numpy as np

# For each query variation's result set:
mean_score = np.mean(scores)
std_score = np.std(scores)

# Z-score normalize each node's score
z_score = (score - mean_score) / std_score if std_score > 0 else 0

# Sigmoid transform into [0, 1]
normalized_score = 1 / (1 + np.exp(-z_score))

# Sum normalized scores for the same node across all query variations
combined_score = sum(normalized_score for each query variation)
```

Process:
1. Calculate mean and standard deviation of scores for each query variation.
2. Z-score normalize: `z = (score - mean) / std`.
3. Sigmoid-transform the z-score into `[0, 1]`.
4. Sum the normalized scores for each node across all query variations.
5. Final ranking reflects each result's statistical significance across all query forms, not just its raw similarity score.

---

## Fusion Mode Comparison

| Mode | Characteristic | Choose When |
|---|---|---|
| **RRF** (`reciprocal_rerank`) | Most robust, rank-based, scale-invariant | You need production stability across heterogeneous retrievers |
| **Relative Score** | Preserves confidence, normalizes by max score | You need score interpretability |
| **Distribution-Based** | Most sophisticated, statistical normalization | You need statistical robustness across many query variations |

---

## Recommended Retrievers by Use Case

| Use Case | Primary Retriever | Enhancement | Why |
|---|---|---|---|
| **General Q&A** | Vector Index Retriever | Combine with BM25 via Query Fusion | Blends semantic relevance with keyword matching |
| **Technical documentation** | BM25 Retriever | Vector Index Retriever as secondary | Prioritizes exact technical terms while keeping semantic flexibility |
| **Long documents** | Auto Merging Retriever | — | Retrieves longer parent chunks only when enough child chunks are matched, preserving context |
| **Research papers** | Recursive Retriever | — | Follows citations/references to pull in relevant cited content |
| **Large document collections** | Document Summary Index Retriever | Followed by Vector Index Retriever | Narrows to relevant documents first, then does detailed retrieval within that subset |

---

## Key Takeaways

- ✅ No single retriever is best for every scenario — retriever choice should match document structure and query pattern.
- ✅ **Vector Index Retriever** is the semantic-search foundation; **BM25** adds precise keyword matching on top of or alongside it.
- ✅ **Auto Merging** and **Recursive** retrievers solve context problems — one via hierarchical parent/child merging, the other via reference/citation traversal.
- ✅ **Document Summary Index Retriever** is a pre-filtering step that keeps fine-grained search fast over large collections.
- ✅ **Query Fusion Retriever** combines multiple retrievers and/or query rewordings into one ranked list — pick RRF for stability, Relative Score for interpretability, or Distribution-Based fusion for statistical robustness.
- ✅ Hybrid combinations (e.g., Vector + BM25 fused with RRF) generally outperform any single retriever for general-purpose Q&A.

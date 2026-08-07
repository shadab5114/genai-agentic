# LangChain Context Retrieval — Cheat Sheet

> **Topic:** Building smarter search over large text collections using LangChain retrievers

---

## Table of Contents

- [Overview](#overview)
- [Why Retrievers Matter](#why-retrievers-matter)
- [Retriever vs. Vector Store](#retriever-vs-vector-store)
- [Building Blocks of a Retriever Pipeline](#building-blocks-of-a-retriever-pipeline)
- [Vector Store-Backed Retriever](#vector-store-backed-retriever)
- [Multi-Query Retriever](#multi-query-retriever)
- [Self-Querying Retriever](#self-querying-retriever)
- [Parent Document Retriever](#parent-document-retriever)
- [Choosing the Right Retriever](#choosing-the-right-retriever)
- [Key Takeaways](#key-takeaways)

---

## Overview

When working with a large collection of text documents — research papers, legal documents, customer service logs, etc. — the goal is often to quickly retrieve the most relevant segments of text based on a user's query. Traditional keyword-based search frequently fails to capture the nuanced meanings and context within documents.

LangChain provides several **retriever** abstractions that go beyond simple keyword search, using embeddings, query rewriting, and document hierarchy to return more relevant results.

The four core retriever types covered here:

1. **Vector Store-Backed Retriever** — semantic similarity search over embedded chunks
2. **Multi-Query Retriever** — generates multiple query variations to widen recall
3. **Self-Querying Retriever** — auto-extracts structured filters from natural-language queries
4. **Parent Document Retriever** — retrieves small chunks but returns their larger parent context

---

## Why Retrievers Matter

| Benefit | Explanation |
|---|---|
| **Efficiency** | Enable fast retrieval of relevant information from large datasets, saving time and compute. |
| **Accuracy** | Advanced retrieval techniques return more contextually relevant results than keyword search. |
| **Versatility** | Different retrievers can be tailored to specific use cases and data/query types. |
| **Context awareness** | Some retrievers (e.g., Parent Document Retriever) consider the broader document context, improving relevance of what's returned to the LLM. |

---

## Retriever vs. Vector Store

A **retriever** is an interface that returns `Document` objects based on an unstructured (natural-language) query. It takes a string in and returns a list of `Document`s.

A **vector store** stores and retrieves documents via embeddings — it's a possible *backend* for a retriever, but not the only one. Retrievers are a thin abstraction that can wrap a vector store's search methods (similarity search, MMR, etc.) or add extra logic on top (query rewriting, metadata filtering, parent-child lookups).

```
query (string) → Retriever → List[Document]
```

---

## Building Blocks of a Retriever Pipeline

Creating a retriever generally involves four steps:

1. **Build the LLM** — a language model used for understanding/generating text (and, for some retrievers, for rewriting or parsing queries).
2. **Split documents into chunks** — break large documents into smaller, manageable pieces so the model can focus on specific sections instead of the whole document.
3. **Build an embedding model** — converts text chunks into numerical vectors representing semantic meaning, enabling similarity comparison.
4. **Retrieve related knowledge from text** — use the retriever interface to fetch the chunks most relevant to a query.

```python
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load the document
loader = TextLoader("data.txt")
txt_data = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(txt_data)
```

---

## Vector Store-Backed Retriever

The simplest retriever type. It's a lightweight wrapper around a vector store that conforms to the retriever interface, using the vector store's own search methods — **similarity search** or **Maximum Marginal Relevance (MMR)** — to query stored text.

**Best for:** semantic similarity and relevance search over large text datasets, when you don't need query rewriting or structured filtering.

```python
from langchain_community.vectorstores import Chroma

# Build a vector store from the chunks
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding_model)

# Direct similarity search
sub_docs = vectordb.similarity_search("smoking policy")
print(sub_docs[0].page_content)

# Or use it as a retriever
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retriever.invoke("smoking policy")
```

MMR can be used instead of plain similarity search to reduce redundancy among the top results:

```python
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
```

---

## Multi-Query Retriever

Query phrasing matters a lot for similarity search — a single phrasing of a question may miss relevant documents phrased differently. The **Multi-Query Retriever** uses an LLM to automatically generate multiple variations of the user's query from different perspectives, runs retrieval for each, and takes the union of the unique results.

**Best for:** situations where a single query phrasing might not capture all relevant documents — improves recall by covering multiple angles on the same question.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(),
    llm=llm
)

docs = retriever.invoke("What are the company's policies on remote work?")
```

Behind the scenes, this generates several reworded queries (e.g., "remote work rules", "work-from-home guidelines", "telecommuting policy"), retrieves for each, and de-duplicates the combined results.

---

## Self-Querying Retriever

Some queries mix **semantic intent** with **structured filters** — e.g., "shows released after 2015 about aliens" combines a semantic search ("aliens") with a metadata filter (`year > 2015`). The **Self-Querying Retriever** uses an LLM to parse the natural-language query into (1) a semantic search string and (2) a structured metadata filter, then applies both against the vector store.

Requires document metadata to be defined with an `AttributeInfo` schema, and the `lark` parsing library to build the structured filter query.

**Best for:** automatically generating and refining queries that combine semantic search with metadata constraints (dates, categories, authors, ratings, etc.).

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

metadata_field_info = [
    AttributeInfo(name="source", description="The document the chunk is from", type="string"),
    AttributeInfo(name="year", description="The year the document was published", type="integer"),
]
document_content_description = "Summary of a document"

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectordb,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
)

docs = retriever.invoke("What documents from 2023 discuss data privacy?")
```

---

## Parent Document Retriever

Splitting documents into very small chunks improves embedding-based similarity matching, but small chunks can lose surrounding context needed for a good answer. Splitting into very large chunks preserves context but hurts retrieval precision. The **Parent Document Retriever** resolves this tension by:

1. Splitting documents into small **child chunks** for embedding and similarity search.
2. Keeping a reference from each child chunk back to its larger **parent chunk/document**.
3. On retrieval, finding the best-matching child chunks, then returning their **parent** documents (or larger parent chunks) instead of the small pieces — preserving full context.

**Best for:** maintaining context and relevance by considering the broader parent document instead of returning isolated fragments.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectordb,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(txt_data)

# Returns full parent chunks, not just the small matching fragment
docs = retriever.invoke("smoking policy")
```

---

## Choosing the Right Retriever

| Retriever | Solves | Requires |
|---|---|---|
| **Vector Store-Backed** | Basic semantic similarity search | Embedding model + vector store |
| **Multi-Query** | Query phrasing sensitivity, low recall from a single phrasing | LLM to generate query variations |
| **Self-Querying** | Queries that mix semantic search with structured metadata filters | LLM + `lark` + defined metadata schema |
| **Parent Document** | Small chunks needed for matching, but larger context needed for answering | Child/parent text splitters + a document store |

> **Note:** These retrievers are not mutually exclusive — in practice, retrieval strategies are often combined (e.g., a Parent Document Retriever whose vector store is queried via Multi-Query rewriting).

---

## Key Takeaways

- ✅ A retriever is an interface that takes a natural-language query and returns relevant `Document`s — it's broader than a vector store, though a vector store often backs it.
- ✅ **Vector Store-Backed Retrievers** are the simplest option — a thin wrapper exposing similarity search / MMR from a vector store.
- ✅ **Multi-Query Retrievers** improve recall by generating multiple phrasings of a query via an LLM and merging results.
- ✅ **Self-Querying Retrievers** let users combine semantic search with structured metadata filters expressed in plain language.
- ✅ **Parent Document Retrievers** balance precise small-chunk matching with the need for full contextual answers by returning the larger parent chunk/document.
- ✅ Chunk size and overlap (e.g., `chunk_size=200`, `chunk_overlap=20`) are key tuning parameters that affect retrieval quality across all retriever types.
- ✅ Choice of retriever should match the query pattern of your use case — pure semantic search, phrasing-sensitive queries, filter + semantic queries, or context-preservation needs.

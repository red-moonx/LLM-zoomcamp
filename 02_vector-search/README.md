# Module 2: Vector search

Notes and implementations for Module 2 of the LLM Zoomcamp covering vector search, embeddings, vector databases, and similarity metrics.

## Overview

In this module, we transition from keyword-based search (like full-text search) to semantic retrieval using vector embeddings and similarity metrics.

### Key concepts

- **Embeddings:** Dense vector representations of text that capture semantic meaning.
- **Vector search & similarity metrics:** Methods like cosine similarity, dot product, and Euclidean distance to compare text semantics.
- **Vector databases & indexing:** Storing and retrieving vector embeddings efficiently at scale.


## What is vector search?

Vector search converts text into numerical arrays (vector embeddings) to retrieve documents based on their underlying semantic meaning rather than exact word matches.

### Simple example

Since computers only understand math, an embedding model maps words to numerical coordinates based on their meaning:
- `"dog"` $\rightarrow$ `[0.85, 0.12, 0.94]` (located very close to `"cat"` or `"pet"`).
- `"airplane"` $\rightarrow$ `[-0.70, -0.99, 0.10]` (located far away).

By measuring the mathematical distance between these coordinates, vector search finds documents that share the same idea, even if they use completely different words.

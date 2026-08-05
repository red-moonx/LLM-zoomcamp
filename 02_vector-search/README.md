# Module 2: Vector search

Notes and implementations for Module 2 of the LLM Zoomcamp covering vector search, embeddings, vector databases, and similarity metrics.

## Overview

In this module, we transition from keyword-based search (like full-text search) to semantic retrieval using vector embeddings and similarity metrics.

### Key concepts

- **Embeddings:** Dense vector representations of text that capture semantic meaning.
- **Vector search & similarity metrics:** Methods like cosine similarity, dot product, and Euclidean distance to compare text semantics.
- **Vector databases & indexing:** Storing and retrieving vector embeddings efficiently at scale.


## What is vector search?

Vector search converts text into numerical arrays (vector embeddings) to retrieve documents based on their underlying semantic meaning rather than exact word matches. For example:

Since computers only understand math, an embedding model maps text to numerical coordinates based on their meaning:
- `"dog"` $\rightarrow$ `[0.85, 0.12, 0.94]` (located very close to `"cat"` or `"pet"`).
- `"airplane"` $\rightarrow$ `[-0.70, -0.99, 0.10]` (located far away).

This process works not just for single words, but for full sentences and entire documents.

In our course RAG pipeline, we convert every FAQ entry into a vector embedding. When a user submits a question, we convert their query into a vector and calculate its similarity against all FAQ vectors to select the **top 5 closest documents** containing the answer.


## Sentence embeddings with SBERT (`sentence-transformers`)

To convert text into vector embeddings, we use **`sentence-transformers`** (SBERT), a popular open-source Python library. SBERT runs locally on your machine (no API costs) and is trained to encode entire sentences into dense vector spaces while preserving contextual meaning.

### Context-aware embeddings

Unlike simple word lookup models, SBERT encodes the entire sentence in context. For example, it understands that the word *"judge"* in a legal context (*"the judge ruled out the possibility"*) has a different meaning and vector than in an ML evaluation context (*"LLM-as-a-judge"*).

### Model selection (`all-MiniLM-L6-v2`)
There are many models  (they can be found in "pretrained_models"). In this course, we are using all-MiniLM-L6-v2 (the smallest and fastest). In the capstone we may need to select a larger one.

We use the `all-MiniLM-L6-v2` model:
- **Vector dimension:** 384 dimensions (compact and fast).
- **Performance:** High speed on CPU, making it lightweight (~80 MB download).
- **Normalized vectors:** Vectors are unit length, meaning the dot product (`v1.dot(v2)`) equals **cosine similarity** directly.

### Basic code example

```python
from sentence_transformers import SentenceTransformer

# 1. Load the pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Encode user query and document into 384-dimensional vectors
query_vector = model.encode("Can I still join the course after the start date?")
doc_vector = model.encode("You can start learning and submitting homework without registering.")

# 3. Calculate similarity (dot product equals cosine similarity for normalized vectors)
similarity_score = query_vector.dot(doc_vector)
print("Similarity score:", similarity_score)
```

### Searching across a dataset (Vector-matrix multiplication)

Rather than looping over documents individually, we use **vector-matrix multiplication** to search the entire dataset at once:

1. **Build a document matrix ($X$):** Stack all $N$ document embeddings into a 2D matrix of shape $(N, 384)$.
2. **Matrix multiplication:** Perform `X.dot(query_vector)` to compute the dot product between the query vector and all $N$ document vectors in a single vectorized operation.
3. **Select top documents:** Sort the resulting scores and select the top $K$ highest scores (e.g., top 5) to retrieve the most relevant FAQ entries for the user.


## RAG with vector search (`RAGVector`)

Because our RAG architecture from Module 1 is modular, swapping keyword search for vector search requires updating only the **search** step. The prompt construction (`build_prompt`) and LLM generation (`llm`) steps remain untouched.

### Subclassing `RAGBase`

In Module 1, we encapsulated the pipeline into `RAGBase` ([rag_helper.py](file:///workspaces/LLM-zoomcamp/01_agentic-rag/rag_helper.py)). `RAGBase.search()` accepts a raw query string for keyword matching. 

For vector search, the index expects a numerical query vector rather than raw text. Instead of rewriting `RAGBase`, we create a subclass named `RAGVector` that:
1. Accepts an `embedder` model (e.g., `all-MiniLM-L6-v2`) in `__init__`.
2. Overrides `search()` to encode the query string into a vector before querying the vector index.

### Why we update the search method

In Module 1, `RAGBase.search()` performed keyword matching directly on text strings. This had a major limitation:

- **Keyword matching failure (Module 1):** If an FAQ document states *"Can I still **join** the course?"* and a user asks *"Is it possible to **enroll** late?"*, keyword search searches for the word `"enroll"`. Because the text uses `"join"`, keyword search scores this document very low or misses it completely.
- **Semantic vector matching (Module 2):** Vector indexes require a numerical vector rather than raw text. Overriding `search()` automatically converts the query string into a vector using `self.embedder.encode()`. Because `"join"` and `"enroll"` map to nearly identical vector coordinates (~0.84 similarity), vector search retrieves the document easily even though `"enroll"` never appeared in the text.

### Implementation (`RAGVector`)

```python
from rag_helper import RAGBase

class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        # 1. Convert query string into a vector embedding
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        # 2. Query the vector index using the vector embedding
        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )
```

### Usage

```python
# Initialize vector assistant
vector_assistant = RAGVector(
    embedder=model,
    index=vindex,
    llm_client=openai_client,
)

# Run full RAG pipeline (vector search -> build prompt -> LLM generation)
answer = vector_assistant.rag("the program has already begun, can I still sign up?")
```

### Why modularity matters

Subclassing `RAGBase` highlights the strength of object-oriented modular design:
- **Search flexibility:** We can swap `minsearch`, `sqlitesearch`, or `PGVector` by overriding `search()`.
- **LLM flexibility:** We can switch LLM providers (e.g., Anthropic, Ollama, OpenAI) by overriding `llm()` without changing search or prompt building logic.


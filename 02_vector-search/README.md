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

## Persistent vector search with sqlitesearch

### Exact vs. approximate nearest neighbors (ANN)

- **Exact NN (k-NN):** Computes similarity against every document vector ($\mathcal{O}(N)$), guaranteeing 100% precision, which is ideal for smaller datasets.
- **Approximate Nearest Neighbors (ANN):** Indexes vectors (e.g., using IVF or HNSW) to search only candidate clusters ($\mathcal{O}(\log N)$), trading a tiny fraction of accuracy for massive speedups at scale.

### Persistent storage with `sqlitesearch`

- **Disk persistence:** Unlike in-memory arrays, `sqlitesearch` saves vector indexes directly to disk (e.g., `faq_vectors2.db`), so embeddings persist across sessions without re-encoding ([vector_search_persistent.ipynb](file:///workspaces/LLM-zoomcamp/02_vector-search/code/vector_search_persistent.ipynb)).
- **ANN indexing mode (`ivf`):** Configures `sqlitesearch` using `mode='ivf'` (Inverted File Index) to partition vector space into clusters for faster query retrieval.
- **Single-notebook pipeline:** Demonstrates both index creation/persistence and running the full `RAGVector` assistant pipeline against SQLite in [vector_search_persistent.ipynb](file:///workspaces/LLM-zoomcamp/02_vector-search/code/vector_search_persistent.ipynb).


## Vector search with pgvector

While lightweight SQLite databases (`sqlitesearch`) work well for local experimentation, production RAG applications often require robust relational databases or enterprise vector stores.

`pgvector` is PostgreSQL with the `vector` extension pre-installed, adding native vector storage, indexing, and similarity search capabilities directly inside a relational PostgreSQL database ([vector_search_pgvector.ipynb](file:///workspaces/LLM-zoomcamp/02_vector-search/code/vector_search_pgvector.ipynb)).

### Production architecture: Ingestion vs. assistant

In real-world production architectures, vector ingestion and assistant retrieval are decoupled into separate concerns:
- **Ingestion pipeline:** Documents are batch-processed, encoded into vectors, and inserted into PostgreSQL as an offline background job.
- **Assistant pipeline:** The live RAG assistant queries the pre-populated `pgvector` store, constructs context prompts, and calls the LLM at runtime.

### Setup and schema initialization

1. **Run PostgreSQL with `pgvector` using Docker:**
   ```bash
   docker run -it \
       --name pgvector \
       -e POSTGRES_USER=user \
       -e POSTGRES_PASSWORD=password \
       -e POSTGRES_DB=faq \
       -p 5432:5432 \
       pgvector/pgvector:pg16
   ```

2. **Enable the vector extension in PostgreSQL:**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Define table schema with a vector column:**
   ```sql
   CREATE TABLE documents (
       id SERIAL PRIMARY KEY,
       course TEXT,
       section TEXT,
       question TEXT,
       answer TEXT,
       embedding vector(384)
   );
   ```

3. **Distance operators in `pgvector`:**
   - `<=>`: Cosine distance. Similarity is computed as `1 - (embedding <=> query_vector)`.
   - `<->`: Euclidean / L2 distance.
   - `<#>`: Negative inner product (dot product).

4. **Inserting documents and vector type casting (`vec_to_str`):**
   PostgreSQL requires vector inputs as formatted string arrays (e.g. `'[0.12,-0.45,...]'`). We use `vec_to_str()` to convert Python vector arrays into strings, and `%s::vector` in SQL to cast the text string into PostgreSQL's internal `vector` column type:

   ```python
   def vec_to_str(vector):
       return '[' + ','.join(str(x) for x in vector) + ']'

   # Insert documents and cast formatted vector strings to vector type
   for doc, vec in zip(documents, vectors):
       conn.execute(
           """
           INSERT INTO documents (course, section, question, answer, embedding)
           VALUES (%s, %s, %s, %s, %s::vector)
           """,
           (doc['course'], doc['section'], doc['question'], doc['answer'], vec_to_str(vec))
       )

   conn.commit()
   ```

### HNSW indexing in `pgvector`

To accelerate vector queries using Approximate Nearest Neighbors (ANN), create an HNSW index with cosine distance operators:

```sql
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops);
```

### Implementation (`RAGPgVector`)

By subclassing `RAGBase`, we override `search()` to execute SQL vector similarity queries directly against PostgreSQL via `psycopg`:

```python
from rag_helper import RAGBase

class RAGPgVector(RAGBase):

    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_str = '[' + ','.join(str(x) for x in query_vector) + ']'

        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()

        return [
            {'course': r[0], 'section': r[1], 'question': r[2], 'answer': r[3]}
            for r in rows
        ]
```

### Usage

```python
# Initialize PostgreSQL vector assistant
vector_assistant = RAGPgVector(
    embedder=model,
    conn=conn,
    llm_client=openai_client,
)

# Run full RAG pipeline using pgvector store
answer = vector_assistant.rag("the program has already begun, can I still sign up?")
```


## Next steps

### When do we actually need vector search?

Adding vector search introduces significant technical overhead (embedding models, vector indexes, specialized database setups, and higher infrastructure complexity). Before adopting vector search for a project or capstone, we must ask ourselves: **Is it really worth it?**

- **Start simple with text search:** Traditional keyword/full-text search (like BM25, SQLite FTS, or `minsearch`) is fast, lightweight, zero-cost, and often fulfills initial product needs completely.
- **When keyword search works best:** If users query using exact terms, product names, error codes, or specific domain jargon, text search performs exceptionally well.
- **Hybrid search (best of both worlds):** Text search and vector search are not mutually exclusive. **Hybrid search** combines keyword retrieval (for exact terms, IDs, and jargon) with vector search (for semantic intent), reranking the combined results. In practice, hybrid search frequently outperforms either approach alone.
- **Evaluating retrieval quality:** How do we know if vector search is actually needed? We determine this through systematic **evaluation** (measuring metrics like Hit Rate or MRR against a ground-truth dataset). We will cover retrieval evaluation in detail in Module 4.
- **Recommended engineering workflow:** Always build a simple keyword search baseline first, evaluate its performance, and only introduce vector search (or hybrid search) when evaluation metrics prove that keyword search alone is insufficient.


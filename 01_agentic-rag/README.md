## Set up the environment
I'm using codespaces. 
I will install uv the requirements as specified here:
https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-agentic-rag/lessons/02-environment.md

```bash
pip install uv
uv init #initializes a project
uv add requests minsearch openai jupyter python-dotenv #installs the requirements
```


## What is RAG?
RAG (Retrieval-Augmented Generation) is a technique that combines the power of Large Language Models (LLMs) with an external retrieval system to improve the accuracy and relevance of generated text. As one of the most common applications of LLMs, RAG allows models to retrieve domain-specific information from external knowledge bases that they were not originally trained on. 

This is especially useful for tasks that require up-to-date information or specialized knowledge, such as question answering over private documents, technical support, or customer service.

By retrieving relevant text passages matching a user's query and passing them as **context** directly into the prompt, RAG grounds the LLM's generation in factual data. This dramatically reduces hallucinations and ensures the model synthesizes accurate answers strictly from the provided source information.

So in RAG we have two parts: first, **retrieval** (R) in which we retrieve relevant documents from a knowledge base, and second, **augmented generation** (AG) in which we use the retrieved documents to generate the answer using LLM.

In this workshop, we start with clean data to focus on the GenAI implementation. In practice, however, a significant portion of RAG development involves data cleaning and preparation.

We have three parts in our pipeline: retrieval (search), building a prompt and generation (LLM).

## 1. Search
Passing all raw data directly into the LLM prompt for every request is computationally expensive, slow, and constrained by context window limits.

Instead, we use dedicated **search engines** or **retrieval systems** (such as **Elasticsearch**, **Apache Solr**, **PostgreSQL (pgvector)**, or lightweight tools like **minsearch**). These systems preprocess and **index** documents ahead of time using techniques like inverted indexes or vector embeddings.

These libraries are somehow heavy too and will require to run docker (therefore it is not convenient to do this in Colab or at least I did not find a way to do it easily). I am going to use minsearch (which is more lightweight) in codespaces.

When a user asks a question, the search engine rapidly identifies and retrieves only the top relevant documents or passages. This keeps the retrieval step fast, precise, and cost-effective before passing the context to the LLM.

### Text fields vs. keyword fields

When defining search indexes, fields are categorized based on how they are stored and queried:

* **`text_fields`** are used to **search** within content (breaking text into tokens for partial and relevance matching).
* **`keyword_fields`** are used to **filter** by exact categories or identifiers (treating values as a single, unbroken code).

#### Example with `minsearch`:

```python
from minsearch import Index

index = Index(
    text_fields=['question', 'section', 'answer'], # Full-text search across content
    keyword_fields=['course']                      # Exact match filtering by course
)
index.fit(documents)
```

In this example:
* `question`, `section`, and `answer` are text fields, allowing full-text search across questions and answers.
* `course` is a keyword field, enabling fast exact filtering (e.g., restricting search results to `course='llm-zoomcamp'`).

## 2. Building a prompt

A prompt is the formatted text string sent to the LLM. In RAG pipelines, it typically combines two main components:

1. **Static system instructions (template):** The fixed rules and persona definition that do not change between requests (e.g., *"Answer the user's question using ONLY the provided context. If missing, say 'I don't know'"*).
2. **Dynamic data (variables):** The variable information injected at runtime for each request:
   - **Retrieved context:** The relevant document snippets returned by the search engine in Step 1.
   - **User question:** The original question submitted by the user.

### Example prompt template:

```python
prompt_template = """
You are a course assistant. Answer the user's QUESTION based ONLY on the provided CONTEXT.
If the answer cannot be found in the context, reply with "I don't know."

Question: {question}

Context:
{context}
""".strip()
```

## 3. Generation (LLM)

Finally, the formatted prompt containing both instructions and retrieved context is sent to the LLM to generate the final response.

### Connecting to OpenAI (`openai_client`)

To interact with OpenAI, we instantiate an `openai_client` object (an instance of the `OpenAI` class). This object manages connection settings, API credentials (API key), and network calls to OpenAI's servers:

```python
from openai import OpenAI

openai_client = OpenAI()
```

### Calling the API (`responses.create` vs `chat.completions.create`)

OpenAI provides two primary SDK methods for text generation:

* **`responses.create` (modern / unified API):** OpenAI's newer, unified interface. Accepts direct inputs and returns cleaner output text (`response.output_text`):
  ```python
  response = openai_client.responses.create(
      model='gpt-5.4-mini',
      input=prompt
  )
  answer = response.output_text
  ```
* **`chat.completions.create` (legacy API):** The traditional endpoint. Requires a list of structured message dicts and nested response parsing (`response.choices[0].message.content`).

### Complete RAG pipeline

We combine retrieval, prompt construction, and generation into clean, reusable functions:

```python
def build_context(search_results):
    lines = []
    for doc in search_results:
        lines.append(doc['section'])
        lines.append('Q: ' + doc['question'])
        lines.append('A: ' + doc['answer'])
        lines.append('')
    return '\n'.join(lines).strip()

def build_prompt(question, search_results):
    context = build_context(search_results)
    return prompt_template.format(question=question, context=context).strip()

def llm(instructions, user_prompt, model='gpt-5.4-mini'):
    message_history = [
        {'role': 'developer', 'content': instructions},
        {'role': 'user', 'content': user_prompt}
    ]
    response = openai_client.responses.create(
        model=model,
        input=message_history
    )
    return response.output_text

def rag(query, model='gpt-5.4-mini'):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    return llm(INSTRUCTIONS, prompt, model=model)
```

Why `rag(query)` is our main orchestrator:
1. **Retrieval (R):** `search(query)` fetches the top relevant documents from `minsearch`.
2. **Augmentation (A):** `build_prompt(...)` combines retrieved context with the user's question.
3. **Generation (G):** `llm(...)` passes developer rules and context to OpenAI to get the final answer.

This packages the whole 3-step pipeline into a single, clean function call: `rag("your question")`.

## 4. Tokens and prompt caching

### How tokens and pricing work
* **Tokens:** LLMs don't read raw words; they process subword chunks called tokens (1 token is roughly 4 characters or 0.75 words).
* **API costs:** OpenAI charges based on `input_tokens` (the prompt and retrieved context you send) and `output_tokens` (the generated answer).

### How prompt caching works (`cached_tokens`)
* **Prefix matching:** When a prompt is longer than 1,024 tokens, OpenAI automatically caches the matching prefix text from left to right across requests.
* **Cost and speed:** Cached input tokens get a 50% discount and respond much faster.
* **Prompt layout tip:** Always put static text (like system instructions and retrieved context) at the top of the prompt and variable text (like the user's question) at the bottom. This prevents breaking the cached prefix on every call.

### Using role-based messages (`message_history`)

Since LLMs are stateless (they do not remember previous requests), we pass the full conversation context on every call using a list of message dictionaries (`message_history`).

```python
message_history = [
    {'role': 'developer', 'content': instructions},  # static system rules
    {'role': 'user', 'content': prompt}               # dynamic user prompt and context
]
```

Why this improves your pipeline:
* **Role hierarchy & authority:** Instructing the model with distinct roles (`developer`/`system`, `user`, `assistant`) ensures the LLM treats developer instructions as high priority.
* **Security (prompt injection prevention):** Explicitly tagging user inputs as `'role': 'user'` prevents malicious user text or retrieved documents from overriding system instructions.
* **Reliable caching:** Keeps the constant system message first so the prefix can be cached easily across calls.
* **Multi-turn chat support:** Enables conversation memory by appending alternating `user` and `assistant` messages to `message_history`.

With this final step, we have implemented all three steps of the RAG pipeline. 

## 5. Modularizing RAG (`ingest.py` and `rag_helper.py`)

To make our RAG pipeline reusable, clean, and extensible (for example, swapping `minsearch` for another search engine without rewriting code), we extract the logic into two Python modules: `ingest.py` and `rag_helper.py`.

### Data ingestion with `ingest.py`

`ingest.py` is dedicated exclusively to data fetching and index preparation. It exposes two core functions:

1. **`load_faq_data()`:** Fetches raw FAQ JSON data from the remote course repository.
2. **`build_index(documents)`:** Initializes a `minsearch.Index` with `text_fields=['question', 'section', 'answer']` and `keyword_fields=['course']`, and fits it with the documents.

These two functions are imported and executed in notebooks (such as [rag_cleaned.ipynb](file:///workspaces/LLM-zoomcamp/01_agentic-rag/rag_cleaned.ipynb)) or application entry points (like Streamlit apps or FastAPI endpoints) to prepare the search index before serving user queries.

### Encapsulation with `rag_helper.py` (`RAGBase`)

Instead of defining scattered functions across notebooks, [rag_helper.py](file:///workspaces/LLM-zoomcamp/01_agentic-rag/rag_helper.py) encapsulates the entire RAG pipeline into a reusable class named `RAGBase`.

#### Why use a class? (Encapsulation and dependency injection)

Using a class provides key architectural advantages:

1. **Dependency injection and state management:** The constructor (`__init__`) receives and stores all essential dependencies as object attributes (`self.xxx`):
   - **`index`:** The fitted `minsearch` index containing the FAQ documents.
   - **`llm_client`:** The `OpenAI` client connection instance.
   - **`model`:** The target LLM model string (defaults to `'gpt-5.4-mini'`).
   - **`instructions`:** System/developer prompt rules enforcing strict context-based answering.
   
   Once injected at instantiation (`assistant = RAGBase(index, client)`), these attributes are preserved across calls without needing to pass them repeatedly.

2. **Clean single-entry interface:** The internal methods (`search`, `build_context`, `build_prompt`, `llm`) handle step-by-step logic, while exposing a simple orchestrator method: `assistant.rag(query)`.

### Example usage

```python
from ingest import load_faq_data, build_index
from rag_helper import RAGBase
from openai import OpenAI

# 1. Load documents and build index (from ingest.py)
documents = load_faq_data()
index = build_index(documents)

# 2. Initialize OpenAI client and RAG helper (from rag_helper.py)
client = OpenAI()
assistant = RAGBase(index=index, llm_client=client)

# 3. Query the pipeline
answer = assistant.rag("When does the course start?")
```

## 6. Data ingestion and persistence

In basic in-memory pipelines (such as our initial setup with `minsearch`), data and search indexes exist solely in the RAM of the Python process.

### Why we need data persistence

Relying on purely in-memory indexing presents two major limitations:

1. **Data volatility (lack of persistence):** Every time the Jupyter notebook kernel restarts, VS Code closes, or the server stops, the in-memory index is cleared. This forces us to re-download raw data and rebuild the entire index from scratch on every run.
2. **Scalability limits:** In-memory indexes cannot scale to production workloads containing millions of documents or large vector embeddings that exceed available RAM.

### Moving to persistent data storage

To solve these limitations, production RAG systems separate the architecture into two distinct pipelines:

* **Ingestion pipeline (one-time or scheduled batch):** A dedicated ingestion script (such as [persistent_rag_ingest.ipynb](file:///workspaces/LLM-zoomcamp/01_agentic-rag/persistent_rag_ingest.ipynb)) processes raw documents and persists the structured index to disk or an external database server.
* **Query pipeline (on-demand):** Applications and notebooks (such as [persinsent_rag.ipynb](file:///workspaces/LLM-zoomcamp/01_agentic-rag/persinsent_rag.ipynb)) connect directly to the existing, pre-built persistent store to serve user queries instantly without re-indexing data.






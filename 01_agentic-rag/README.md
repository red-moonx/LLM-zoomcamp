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

When a user asks a question, the search engine rapidly identifies and retrieves only the top relevant documents or passages. This keeps the retrieval step fast, precise, and cost-effective before passing the context to the LLM.

### Text Fields vs. Keyword Fields

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

## 2. Building a Prompt

A prompt is the formatted text string sent to the LLM. In RAG pipelines, it typically combines two main components:

1. **Static System Instructions (Template):** The fixed rules and persona definition that do not change between requests (e.g., *"Answer the user's question using ONLY the provided context. If missing, say 'I don't know'"*).
2. **Dynamic Data (Variables):** The variable information injected at runtime for each request:
   - **Retrieved Context:** The relevant document snippets returned by the search engine in Step 1.
   - **User Question:** The original question submitted by the user.

### Example Prompt Template:

```python
prompt_template = """
You are a course assistant. Answer the user's QUESTION based ONLY on the provided CONTEXT.
If the answer cannot be found in the context, reply with "I don't know."

Question: {question}

Context:
{context}
""".strip()
```

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

### Complete RAG pipeline

We combine retrieval, prompt construction, and generation into a single end-to-end function:

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

def rag(question):
    search_results = search(question)
    prompt = build_prompt(question, search_results)
    return llm(prompt)
```

With `rag(question)`:
1. **Retrieval (R):** `search(question)` fetches the top matching documents from `minsearch`.
2. **Prompt building:** `build_prompt(...)` formats the documents and question into context.
3. **Augmented generation (AG):** `llm(prompt)` sends the complete context-grounded prompt to OpenAI for the final answer.

## 4. Tokens and prompt caching

### How tokens and pricing work
* **Tokens:** LLMs don't read raw words; they process subword chunks called tokens (1 token is roughly 4 characters or 0.75 words).
* **API costs:** OpenAI charges based on `input_tokens` (the prompt and retrieved context you send) and `output_tokens` (the generated answer).

### How prompt caching works (`cached_tokens`)
* **Prefix matching:** When a prompt is longer than 1,024 tokens, OpenAI automatically caches the matching prefix text from left to right across requests.
* **Cost and speed:** Cached input tokens get a 50% discount and respond much faster.
* **Prompt layout tip:** Always put static text (like system instructions and retrieved context) at the top of the prompt and variable text (like the user's question) at the bottom. This prevents breaking the cached prefix on every call.

### Using role-based messages
Instead of sending a single plain string to the API, you can structure your prompt as a list of message roles:

```python
message_history = [
    {'role': 'developer', 'content': instructions},  # static system rules
    {'role': 'user', 'content': prompt}               # dynamic user prompt and context
]
```

Why this improves your pipeline:
* **Better instruction compliance:** Separating system rules from user input prevents user text or retrieved documents from overriding your instructions.
* **Reliable caching:** Keeps the constant system message first so the prefix can be cached easily across calls.
* **Chat support:** Makes it simple to support multi-turn conversations by just appending `assistant` and `user` follow-ups.





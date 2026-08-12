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

LLMs are only trained up to a specific knowledge cutoff date. While modern models can use live web search to fetch real-time information, web search alone is not a guarantee of accuracy because the model retrieves unverified information from across the web. Therefore, providing high-quality, targeted context is essential—LLMs perform only as well as the context they receive.

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

Essentially, we are separating the ingestion from the RAG assistant.

## SQLite search (`sqlitesearch` and FTS5)


To achieve data persistence, we replace `minsearch` with `sqlitesearch`, a Python library built on top of SQLite.

### Key concepts

* **SQLite:** A lightweight, serverless database storing all data locally in a single file (`faq.db`), ensuring data persists across application restarts.
* **`sqlitesearch`:** A Python wrapper offering a simple `minsearch`-like interface (`TextSearchIndex`) while storing records directly in SQLite.
* **FTS5 (Full-Text Search 5):** A native SQLite extension that creates an inverted index of text fields (similar to index pages at the back of a book), enabling high-speed keyword queries across thousands of documents.

### Comparison: `minsearch` vs. `sqlitesearch`

| Feature | `minsearch` | `sqlitesearch` |
| :--- | :--- | :--- |
| **Storage location** | Volatile RAM (cleared on kernel restart) | Persistent disk file (`faq.db`) |
| **Search engine** | Custom in-memory Python index | Native SQLite FTS5 engine |
| **Primary benefit** | Fast and zero-dependency setup | Permanent data storage & idempotency |

## Agents

Standard RAG follows a rigid, single-pass pipeline: **Retrieve $\rightarrow$ Prompt $\rightarrow$ Generate**. This flow leaves little room for recovery—if a query contains a typo or poor phrasing, keyword search may fail, leading to poor model answers. Agents solve this by introducing dynamic control flow, allowing the system to evaluate results, reformulate queries, and self-correct.

### Function calling

Using agents allows for greater flexibility by putting the LLM in charge. Instead of manually running search queries ourselves, we provide the LLM with a search tool so it can decide when to execute a search and what parameters to use.

#### 1. Defining the search function

First, we define a top-level search function that queries the index directly. The model will reference this function by name, so keeping the Python function name aligned with the tool name makes dispatching easier later on:

```python
def search(query):
    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
        filter_dict=filter_dict
    )
```

#### 2. Defining the tool schema

Next, we define the tool schema for the model. The model does not see Python code—it only receives a schema describing what the function does and what arguments it accepts. Because LLMs are language-agnostic and API calls use HTTP, tools are described using JSON rather than Python code (the same schema would work in TypeScript or Java):

```python
search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}
```

The `description` field is critical because the model reads it to determine when the tool should be invoked. The `parameters` field follows JSON schema specifications for the arguments, and `query` is marked as required so the model always provides a search string. This tool schema is passed to the client via the native `tools` parameter (e.g., `tools=[search_tool]`), allowing the API to evaluate when to request function calls.

### Execution flow and state management

In this process of transforming standard RAG into agentic RAG, we make two calls to the LLM. Here is a summary of the process:

1. Make a call to the LLM (first call).
2. The LLM decides to invoke `search('params')`.
3. We execute the search and obtain the results.
4. Send the results back to the LLM (second call).
5. The LLM processes the results.
6. The LLM generates the final answer.

Because LLMs are stateless, we must send the entire conversation history to the model before making the second call.

First, we define `function_call_output`, which is the structured object where we package the execution result of our local function along with its matching request ID to send back to the LLM:

```python
function_call_output = {
    "type": "function_call_output",
    "call_id": call.call_id,
    "output": result_json,
}
```

`result_json` is the transformation of the search results into JSON format, which is readable by both humans and machines.

Then, we append both the model's tool call request (`messages.append(call)` or `messages.extend(response.output)`; step 2) and the tool execution results (`messages.append(function_call_output)`; step 3) to the message history so the model has complete context for the second call (step 4). Note that because we make two API calls, we pay for input/output tokens on both step 1 and step 4.

### The agentic loop

In standard function calling, we make a fixed number of API calls. However, complex user queries may require multiple searches with different keywords before enough information is gathered to provide a complete answer. An agent solves this by running an autonomous loop.

#### Multi-call execution flow

1. Make the initial call to the LLM (first call).
2. The LLM decides to invoke `search('params')`.
3. We execute the search and obtain the results.
4. Send the results back to the LLM (second call).
5. The LLM processes the retrieved search results.
6. If information is missing, the LLM decides to make another tool call with refined keywords.
7. We execute the second search and send results back (third call).
8. The LLM processes the new results and generates the final answer.

Because we do not know in advance how many searches the model will need, we run a `while True` loop that executes tool calls until the LLM decides it has enough context to answer.

#### Guiding the model with instructions

We explicitly instruct the model to perform multiple searches in the system prompt:
> *"Make multiple searches. First perform search, analyze the results, and then perform more searches using refined keywords if needed."*

#### Tool dispatching with `make_call`

To handle tool execution dynamically within the loop, we use a helper function named `make_call`:

```python
def make_call(call):
    args = json.loads(call.arguments)

    if call.name == "search":
        result = search(**args)

    result_json = json.dumps(result, indent=2)

    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result_json,
    }
```

`make_call` acts as a dispatcher: it parses the JSON arguments from the LLM, executes the native `search()` function (which queries the top 5 results with field boosting), and formats the output into a JSON object matching `call_id`.

#### Implementation of the loop

```python
def agent_loop(instructions, question, model='gpt-5.4-mini') -> str:
    messages = [
        {'role': 'developer', 'content': instructions},
        {'role': 'user', 'content': question}
    ]

    it = 1
    while True:
        print(f'iteration #{it}...')
        has_function_calls = False

        response = openai_client.responses.create(
            model=model,
            input=messages,
            tools=[search_tool]
        )

        messages.extend(response.output)

        for item in response.output:
            if item.type == 'function_call':
                call_output = make_call(item)
                messages.append(call_output)
                has_function_calls = True

            elif item.type == 'message':
                final_answer = item.content[0].text

        it += 1
        if not has_function_calls:
            break

    return final_answer
```

#### How stopping works

The termination of the loop relies on a combination of LLM decision-making and Python control flow:

1. **LLM evaluation:** On each turn, the model evaluates whether the retrieved results in `messages` answer all parts of the user query. If information is still missing, it outputs another `function_call` item with updated search terms. Once it has enough information, it stops requesting tool calls and outputs a standard text `message`.
2. **Python termination:** Python tracks `has_function_calls`. When the LLM outputs no tool calls (`has_function_calls == False`), Python executes `break` to exit the loop.

## Agent frameworks (ToyAIKit)

While writing the agentic loop from scratch (`while True`, tool dispatchers, manual JSON schemas) provides a clear understanding of how agents work under the hood, building production applications usually relies on agent frameworks. `ToyAIKit` is a lightweight framework used in this module to abstract away this boilerplate. For production systems, frameworks like **LangGraph** (for stateful graph-based orchestration and human-in-the-loop workflows), **CrewAI**, **Smolagents**, and **PydanticAI** are common industry alternatives.

### Key abstractions in `ToyAIKit`

1. **Automatic tool schema generation (`Tools`):**  
   Instead of manually writing JSON schemas (`search_tool`), `Tools.add_tool(search)` inspects Python type hints and docstrings to automatically generate the JSON schema expected by the API.

2. **Runner orchestration (`OpenAIResponsesRunner`):**  
   Encapsulates the `while True` loop, tool routing, history management, and response handling into a single runner instance.

3. **Rich UI callbacks (`IPythonChatInterface` & `DisplayingRunnerCallback`):**  
   Provides interactive rendering in Jupyter notebooks (e.g. collapsible HTML views showing tool calls, inputs, and outputs).

### Registering tools

The first thing we need to do is to register the tools (i.e., adding tools to the agent's toolbox) so the framework knows the tool is available for the LLM to call:

```python
agent_tools = Tools()
agent_tools.add_tool(search, search_tool)
```

We register our search function along with the explicit schema from earlier lessons.

### Schema generation

Writing JSON schemas by hand for every function is verbose. ToyAIKit allows us to skip manual schema creation by automatically generating the schema from Python **type hints** and **docstrings**:

```python
def search(query: str) -> dict[str, str]:
    """
    Search the FAQ database for entries matching the given query.
    """
    return index.search(
        query,
        num_results=5,
        boost_dict={"question": 3.0, "section": 0.5},
        filter_dict={"course": "llm-zoomcamp"}
    )

agent_tools = Tools()
agent_tools.add_tool(search)
```

#### How ToyAIKit derives the schema

1. **Type hints:** Specifies input parameter and return types (e.g., `query: str`). ToyAIKit inspects these hints to set parameter data types like `"type": "string"` in the JSON schema.
2. **Docstrings:** The documentation string inside triple quotes (`"""..."""`). ToyAIKit extracts this text to populate the `"description"` field, telling the LLM what the tool does and when to call it.

The output is the same JSON schema we hand-wrote in the function calling lesson. ToyAIKit generated it from the docstring and the type hint.

Every modern agent framework does this same trick. It reads a typed Python function with a docstring and builds the schema from it. The OpenAI Agents SDK, PydanticAI, LangChain and Google ADK all work this way. You write the tool and the framework figures out how to describe it.

### Chat interface and runner

In simple terms, this component builds a custom, interactive ChatGPT-style assistant connected to our private FAQ dataset:

- **`IPythonChatInterface` (UI layout):** Displays a clean chat interface inside Jupyter notebooks with collapsible view blocks for tool calls.
- **`DisplayingRunnerCallback` (Live listener):** Renders updates in the notebook in real time as the agent thinks, invokes tools, and processes search results.
- **`OpenAIResponsesRunner` (Automated loop):** Replaces our manual `while True` loop. It links the tools, instructions, UI, and LLM into an automated engine that handles tool execution and response cycles until the answer is complete.
- **`OpenAIClient(model="gpt-5.4-mini")`:** Explicitly sets a model with strong instruction-following and tool-dispatching capabilities.

```python
chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)

runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=instructions,
    chat_interface=chat_interface,
    llm_client=OpenAIClient(model="gpt-5.4-mini")
)
```

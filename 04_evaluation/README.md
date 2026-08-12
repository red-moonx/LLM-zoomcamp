# Module 4: Evaluation

Notes and code for Module 4 of the LLM Zoomcamp covering evaluation metrics, retrieval evaluation, LLM response evaluation, and monitoring/testing RAG pipelines.

## Overview

Evaluating RAG pipelines ensures that both the retrieval component (finding relevant context) and the generation component (producing accurate LLM responses) perform reliably.

## Key topics

- **Retrieval evaluation:** Measuring search quality using metrics like Hit Rate, MRR (Mean Reciprocal Rank), precision, and recall.
- **Generation evaluation:** Assessing answer quality using LLM-as-a-judge, ROUGE/BLEU scores, semantic similarity, and hallucination detection.
- **Offline vs online evaluation:** Setting up benchmark evaluation datasets versus monitoring live user feedback.

## Evaluation lifecycle: intro

The evaluation system measures both search quality and the end-to-end pipeline (verifying whether answers are accurate for users and agents).

To continuously improve the system, we collect real interaction data and run an iterative evaluation loop:

1. Interact with the system (ask real user questions)
2. Collect logs (record queries, retrieved contexts, and outputs)
3. Build a test dataset from real questions
4. Generate ground truth and synthetic data
5. Evaluate the pipeline (retrieval + generation metrics)
6. Tune and improve the pipeline (adjust chunking, prompts, or weights)
7. Monitor the production pipeline
8. Set up alerts for regressions or anomalous behavior

Repeat the cycle as new user queries and edge cases arrive.

## Ground-truth generation and search evaluation

We can use an existing FAQ dataset or document collection to generate synthetic user questions. From a single document $D_i$, an LLM can generate multiple plausible user queries $(Q_{i1}, Q_{i2}, \dots, Q_{ik})$, forming query-document pairs $(Q_{ij}, D_i)$.

- **Unique document IDs:** Every document or chunk in the dataset must have a unique identifier (`doc_id`). In evaluation, we use `doc_id` to verify if the search engine successfully retrieves the source document $D_i$ for query $Q_{ij}$.
- **Evaluating search vs. generation:**
  - **Search evaluation:** Send synthetic query $Q_{ij}$ to the search index and check if source document $D_i$ is retrieved in the top results (calculating metrics like Hit Rate and MRR).
  - **Generation evaluation:** Send query $Q_{ij}$ through the full RAG pipeline and check if the generated response matches original answer $A_i$ or satisfies an LLM judge.

> [!TIP]
> **Capstone tip:** Always assign a clean, unique ID (such as a string hash of content + metadata) to every document or chunk. This makes automated retrieval evaluation fast and reproducible.

### Structured output with Pydantic

When generating synthetic questions, we use **structured output** to force the LLM to return data in a deterministic schema instead of unstructured free text.

```python
from pydantic import BaseModel

class Questions(BaseModel):
    questions: list[str]
```

- **`BaseModel` schema:** Defines a Pydantic model with a `questions` field expecting a list of strings (`list[str]`).
- **Enforcing JSON output:** Passing `text_format=Questions` to the API converts this class into a JSON schema, forcing the LLM to output valid JSON.
- **Direct parsing:** Python automatically parses the LLM response directly into a `Questions` object (`response.output_parsed.questions`), eliminating manual string cleaning or regex parsing.

Once we generated a set of synthetic questions, we can use them to evaluate the retrieval component. To do so, we send them to LLM. We have to do this process for the entire dataset (for all items in the dataset). 

We use a function from evaluation_utils.py for this process:
`llm_structured` is a helper function that prompts an LLM to return data in a strict Pydantic schema, returning both the parsed result and token usage metrics.

## Ground-truth for all documents

When generating synthetic questions across an entire dataset, batch processing requires resilient API calls to handle temporary network glitches or rate limits.

We use helper functions from `evaluation_utils.py`:

**`llm_structured_retry`:** A wrapper around `llm_structured` that automatically retries the API call up to $N$ times with exponential backoff if a temporary network or API error occurs.

### Single document generator function

`generate_ground_truth(doc)` converts a single document into ground-truth evaluation pairs:

```python
def generate_ground_truth(doc):
    user_prompt = json.dumps(doc)

    out, usage = llm_structured_retry(
        openai_client,
        data_gen_instructions,
        user_prompt,
        Questions
    )

    results = []
    for q in out.questions:
        results.append({
            "question": q,
            "document": doc["id"]
        })

    return results, usage
```

- **Document serialization:** Converts `doc` to a JSON string (`user_prompt`).
- **Resilient structured call:** Uses `llm_structured_retry()` to generate questions matching the `Questions` Pydantic schema.
- **ID pairing:** Pairs each generated question string `q` with `doc["id"]` for retrieval evaluation.

### Batch processing with tqdm

We iterate through documents using `tqdm` to monitor progress and aggregate generated data:

```python
from tqdm.auto import tqdm

ground_truth = []
usages = []

for doc in tqdm(documents[:5]):
    records, usage = generate_ground_truth(doc)
    ground_truth.extend(records)
    usages.append(usage)
```

- **`tqdm` progress bar:** Displays a progress bar during batch processing.
- **`ground_truth.extend(records)`:** Appends all generated `(question, doc_id)` pair dicts into a flat master list.
- **`usages.append(usage)`:** Accumulates token usage objects to track total API cost.



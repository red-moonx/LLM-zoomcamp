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

### Persisting the ground-truth dataset

After generating question-document pairs, we convert the dataset into a Pandas DataFrame and save it as a CSV file (`data/ground_truth-new.csv`).

```python
import pandas as pd

df_ground_truth = pd.DataFrame(ground_truth)
df_ground_truth.to_csv("data/ground_truth-new.csv", index=False)
```

**Why persist the ground truth to a CSV?**

1. **Cost and speed efficiency:** Generating ground truth requires multiple LLM API calls, which consume time and credits. Saving to CSV allows downstream evaluation steps (`02-search-eval.ipynb`, `03-rag-evals.ipynb`, `04-llm-judge.ipynb`) to load the benchmark dataset instantly without re-generating questions.
2. **Reproducibility and baseline consistency:** Having a fixed evaluation benchmark ensures fair comparison when testing different retrieval algorithms (text search vs vector search), prompt templates, or chunking strategies.
3. **Offline evaluation:** Allows team members or benchmark runs to test search engines offline without needing live OpenAI API keys for dataset creation.

## Search evaluation

This work is covered in `02-search-eval.ipynb`. For all documents, we want to verify whether the generated question `q` successfully retrieves the source document `d` (`doc_id`) it was created from.

### Verifying document matches

To check if a search engine retrieves the expected source document, we compare the ID of each search result against the ground-truth `doc_id`:

```python
for d in results:
    print(f'{d["id"]} == {doc_id}: {d["id"] == doc_id}')
```

**Simple explanation:**
This loop iterates through the top search results (`results`) returned for a query and compares each document's ID (`d["id"]`) with the expected source document ID (`doc_id`). It prints `True` if a result matches the source document and `False` if it does not.

### The relevance matrix

Converting the boolean check (`d["id"] == doc_id`) into binary values (`1` for match, `0` for non-match) across all test queries produces a **relevance matrix** (`relevance_total`):

```python
relevance = []
for d in results:
    relevance.append(int(d["id"] == doc_id))
```

**Understanding rows and columns:**
- **Each row = one generated test question:** Row 0 is Generated Question 1, Row 1 is Generated Question 2, etc. (synthesized from source documents by the LLM).
- **Each column = search rank position:** Column 0 is the 1st search result, Column 1 is the 2nd result, up to top-N.
- **At most one `1` per row:** Because each generated question comes from a single unique source document (`doc_id`), only one retrieved document can match. If the correct source document is not in the top-N results, the row is all zeros (`[0, 0, 0, 0, 0]`).

```python
# Example matrix for top-5 results across 4 generated queries
relevance_total = [
    [1, 0, 0, 0, 0],  # Generated Question 1: correct document at rank 1
    [0, 1, 0, 0, 0],  # Generated Question 2: correct document at rank 2
    [0, 0, 0, 0, 0],  # Generated Question 3: correct document not in top-5
    [0, 0, 1, 0, 0]   # Generated Question 4: correct document at rank 3
]
```

### Search evaluation metrics

Once we have the relevance matrix (`relevance_total`), we compute two fundamental retrieval evaluation metrics:

#### 1. Hit Rate (Recall@N)

- **Concept:** Measures whether the search engine successfully retrieved the target document anywhere in the top-N results.
- **Interpretation:** High Hit Rate means your search engine rarely misses the correct document completely.
- **Calculation:** Percentage of matrix rows that contain at least one `1` (`True in row`).

$$\text{Hit Rate} = \frac{\text{Number of queries where target doc was retrieved}}{\text{Total number of queries}}$$

```python
def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt += 1
    return cnt / len(relevance_total)
```

#### 2. Mean Reciprocal Rank (MRR)

- **Concept:** Measures how high up in the result list the target document appears. It penalizes systems that return the target document at lower rank positions (e.g. rank 5 vs rank 1).
- **Reciprocal Rank score:** $1 / \text{rank}$ for the position of the `1` in a row (rank 1 $= 1.0$, rank 2 $= 0.5$, rank 3 $= 0.33$, no match $= 0.0$).
- **Calculation:** Average reciprocal rank across all evaluation queries.

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

```python
def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank, val in enumerate(line):
            if val == 1:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)
```
mrr is a better metric to optimize for than hit rate because it rewards systems that return the target document at higher rank positions.

These metrics are based on synthetic data, which is the first step to understand if our search pipeline is working properly. For a final answer and for information, we will need to gather real user data and validate our metrics against it.

## Search parameter tuning
Now that we have our framework for evaluating, we can start tuning our search parameters.

For example, here we are evaluating different boosting parameters for questions

for boost in [0.5, 1.0, 3.0, 5.0, 10.0]:
    result = evaluate(
        ground_truth,
        lambda query, boost=boost: search_boost(query, boost)
    )
    print(f"boost={boost}: {result}")

and found that that boosting with 0.5 is better than boosting with 3. The best result (at the level of mrr) is with not boosting. 

We can do this for all parameters and combinations: grid search.

## RAG and agent evaluation
This section is based on the script: 03-rag-evals.ipynb
To evaluate end-to-end RAG quality, we process each ground-truth question, generate an answer using the RAG pipeline, and pair it with the original reference answer for comparison:

```python
def generate_rag_answer(rec):
    question = rec["question"]
    doc_id = rec["document"]
    original_doc = doc_idx[doc_id]

    answer_llm = assistant.rag(question)
    answer_orig = original_doc["answer"]

    result = {
        "question": question,
        "answer_llm": answer_llm,
        "answer_orig": answer_orig,
        "document": doc_id,
    }

    return result
```

- **`assistant.rag(question)`:** Runs retrieval search, builds context, and generates the LLM answer (`answer_llm`).
- **`doc_idx[doc_id]`:** Looks up the original ground-truth FAQ document to extract its reference answer (`answer_orig`).
- **Comparison record:** Pairs `answer_llm` alongside `answer_orig` so an LLM judge can evaluate response quality.
- **`RAGWithUsage`:** A subclass of `RAGBase` that tracks token usage and costs during batch evaluation.

### LLM as judge

This section is based on `04-llm-judge.ipynb` (and documented in `13-llm-as-judge.md`).

The goal is offline evaluation of the full RAG pipeline ($A \to Q \to A'$), comparing the generated answer ($A'$) against the original ground-truth FAQ answer ($A$) for question ($Q$).

### Key concepts

- **Structured schema (`AnswerEvaluation`):** Uses Pydantic to enforce structured output containing `score` (`"good"` or `"bad"`) and `reasoning` (explanation of the verdict).
- **Evaluation prompt:** Combines system instructions (`aqa_judge_instructions`) with formatted data (`aqa_judge_prompt`) comparing the question, ground truth, and AI answer.
- **Parallel execution & cost:** Evaluates all records in parallel with `ThreadPoolExecutor` and tracks token usage cost via `calc_total_price`.
- **Results & error analysis:** 
  - Achieved ~96% good answers (379/395) at a cost of ~$0.25 USD.
  - Answer accuracy (~96%) exceeded search Hit Rate (~85%) because the LLM can often answer correctly even with imperfect search context.
  - Inspecting `"bad"` cases helps identify search misses, omitted details, or unnecessary fallback responses.
- **Persisting results:** Saves evaluations to `data/rag-evaluations-new.csv`.

## Agent evaluation

This section is based on `04-agent-evaluation.ipynb` (and documented in `14-agent-evaluation.md`).

The goal is offline evaluation of an autonomous AI agent, assessing both the final answer quality and the execution trajectory (`tool_calls`).

### Key concepts

- **Trajectory capturing (`tool_calls`):** Stores the list of tool invocations (function name and arguments) made during the agent loop (`runner.loop`).
- **Dual evaluation schema (`AgentEvaluation`):** Uses Pydantic to enforce two independent scores:
  - `answer_score` (`"good"` / `"bad"`) + `answer_reasoning`: Evaluates semantic correctness of the final response.
  - `trajectory_score` (`"good"` / `"bad"`) + `trajectory_reasoning`: Evaluates relevance, query refinement, and lack of redundant tool calls.
- **Judge prompt:** Combines system instructions (`agent_judge_instructions`) with data (`agent_judge_prompt`) passing the question, ground truth, agent answer, and formatted `tool_calls`.
- **Diagnostic matrix:**
  - `answer` **good** + `trajectory` **good**: Ideal autonomous execution.
  - `answer` **good** + `trajectory` **bad**: Inefficient execution (redundant or poor search queries) that got lucky.
  - `answer` **bad** + `trajectory` **good**: Synthesis failure (retrieved correct context but LLM misread or hallucinated).
  - `answer` **bad** + `trajectory` **bad**: Total failure (wrong search terms, unhelpful context, wrong answer).
- **Benchmark results (50 questions):** Achieved 45/50 good answers (~90%) and 49/50 good trajectories (~98%) at a total cost of ~$0.07 USD for agent execution + ~$0.05 USD for LLM judging.
- **Persisting results:** Saves evaluations to `data/agent-evaluations.csv`.


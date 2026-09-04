# Module 5: Monitoring

Offline evaluation can't tell you how your RAG system performs once real
people use it. This module covers online monitoring: collecting metrics
from real traffic and visualizing them on a dashboard.

We build a Streamlit chat app, capture metrics, store conversations
in PostgreSQL, and create Grafana dashboards for real-time monitoring.

## Lessons

Work through them in order:

1. [Intro](lessons/01-intro.md) - Why monitoring matters, what we'll build
2. [Assistant Setup](lessons/02-assistant-setup.md) - Setting up the RAG assistant
3. [Chat App](lessons/03-chat-app.md) - Basic Streamlit app with RAG
4. [Capturing Metrics](lessons/04-metrics.md) - LLMCallRecord, cost tracking
5. [Database](lessons/05-database.md) - PostgreSQL with Docker, saving conversations
6. [Querying Data](lessons/06-querying.md) - Fetching stored conversations
7. [Streamlit Dashboard](lessons/07-streamlit-dashboard.md) - Visualizing metrics in Streamlit
8. [User Feedback](lessons/08-user-feedback.md) - Thumbs up/down buttons
9. [Built-in Judge](lessons/09-built-in-judge.md) - LLM-as-a-judge for automatic relevance evaluation
10. [Feedback Dashboard](lessons/10-feedback-dashboard.md) - Adding feedback panels to the Streamlit dashboard
11. [Synthetic Data](lessons/11-synthetic-data.md) - Generating test data for dashboards
12. [Grafana Dashboards](lessons/12-grafana.md) - SQL queries and dashboard panels
13. [Docker Compose](lessons/13-docker-compose.md) - Running everything together
14. [Next Steps](lessons/14-next-steps.md) - OpenTelemetry, alerting, frameworks to learn more

## 1. Assistant setup

The assistant is a Python object (an instance of `RAGBase`) that encapsulates the entire RAG pipeline: it loads the course FAQ dataset into a `minsearch` in-memory search index, connects to the OpenAI API, and exposes a `rag(query)` method that executes the 3-step loop (search context $\to$ construct prompt $\to$ generate LLM response). Calling `create_assistant()` in `assistant.py` initializes this pipeline so it can be tested from the command line before wrapping it in a web UI or monitoring database.

## 2. Chat app

The command-line assistant is wrapped into a lightweight web interface using Streamlit in `app.py`. It renders a text input field and submit button, forwarding user queries to `assistant.rag(user_input)` and displaying the generated answer. Running `make chat` launches `uv run streamlit run app.py` to serve the interactive web app locally or via GitHub Codespaces port forwarding.

## 3. Capturing metrics

To instrument the RAG pipeline and eliminate black-box execution, `metrics.py` defines an `LLMCallRecord` dataclass to hold call metadata (prompt, answer, token usage, response latency, cost, timestamp). The `RAGWithMetrics` subclass extends `RAGBase`, overriding the `llm()` method to measure execution time (`time.time()`), compute monetary cost via `calculate_cost()`, and stash the recorded metrics in `self.last_call`. Updating `assistant.py` to use `RAGWithMetrics` allows `app.py` to display latency, token consumption, and cost directly in the Streamlit UI.






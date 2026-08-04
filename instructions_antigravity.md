# Antigravity Context & User Preferences

## Project Overview
This repository contains work for the **DataTalksClub LLM Zoomcamp** course.
Currently, we are working on **Module 1: Agentic RAG / Basic RAG Pipeline** located in `01_agentic-rag/`.

### Key Files in `01_agentic-rag/`
- **[README.md](file:///workspaces/LLM-zoomcamp/01_agentic-rag/README.md):** Complete study notes and explanations covering:
  - Search / Retrieval using `minsearch` (`text_fields` vs `keyword_fields`).
  - Prompt construction & context building.
  - LLM generation (`openai_client`, `responses.create` vs `chat.completions.create`).
  - Tokens, pricing, and prompt caching (`cached_tokens`).
  - Role-based messaging (`developer`, `user`, `assistant` roles in `message_history`).
  - Modularizing RAG (`ingest.py` & `rag_helper.py`).
- **[ingest.py](file:///workspaces/LLM-zoomcamp/01_agentic-rag/ingest.py):** Functions to fetch FAQ JSON data (`load_faq_data`) and build a `minsearch` index (`build_index`).
- **[rag_helper.py](file:///workspaces/LLM-zoomcamp/01_agentic-rag/rag_helper.py):** Class `RAGBase` encapsulating search, prompt building, LLM execution, and the orchestrator `rag(query)`.
- **[notebook.ipynb](file:///workspaces/LLM-zoomcamp/01_agentic-rag/notebook.ipynb):** Interactive Jupyter notebook used during lessons.

---

## User Communication & Tone Preferences

1. **Natural Capitalization (No Random Title Case):**
   - Write like a human developer.
   - Use standard natural sentence case for bullet points, headers, and explanations. Do NOT capitalize every word in bullet titles or section headers unless proper nouns require it.

2. **Tone & Style:**
   - Keep explanations direct, clear, friendly, and natural (avoid robotic or overly corporate AI phrasing).
   - Respond in Spanish when the user asks questions in Spanish, or English if they write in English.

3. **Documentation Structure:**
   - Integrate new notes, summaries, or explanations into the natural narrative flow of `README.md` or code files.
   - Avoid attaching random, disconnected summary blocks at the end of files.

4. **Code & API Conventions:**
   - `openai_client` is a connection object (instance of `OpenAI`).
   - Prefer modern `responses.create` API over legacy `chat.completions.create` where applicable.
   - Maintain the separation of roles (`developer`, `user`, `assistant`) for clarity, security (preventing prompt injection), and prefix caching.

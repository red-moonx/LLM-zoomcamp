import streamlit as st
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv("../.env")

# Configuration
ES_URL = "http://localhost:9200"
INDEX_NAME = "athletica_chunks"
CHUNKS_PATH = "data/processed/optimizations/optimization_3/chunks.json"
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
LLM_MODEL = "gpt-4o-mini"

# Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Athletica Assistant",
    page_icon="🏃‍♀️",
    layout="centered"
)

# --- Initialization & Caching ---
@st.cache_resource(show_spinner="Loading Embeddings Model...")
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner="Connecting to Database...")
def get_elasticsearch_client():
    return Elasticsearch(ES_URL)

@st.cache_resource
def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c for c in chunks}

# Initialize globals
try:
    model = load_embedding_model()
    es = get_elasticsearch_client()
    chunk_dict = load_chunks()
    openai_client = OpenAI()
except Exception as e:
    st.error(f"Error initializing services: {e}")
    st.stop()

# --- Functions ---
def search_knn(query: str, top_k: int = 5):
    """Retrieve chunks using KNN vector search."""
    vector = model.encode(query).tolist()
    search_body = {
        "knn": {
            "field": "dense_vector",
            "query_vector": vector,
            "k": top_k,
            "num_candidates": 100
        }
    }
    response = es.search(index=INDEX_NAME, body=search_body, size=top_k)
    return [hit["_source"]["chunk_id"] for hit in response["hits"]["hits"]]

def generate_answer(question, context):
    """Generate answer using OpenAI."""
    prompt = f"""
You are Athletica, an expert sports physiology and nutrition assistant for female athletes. 
Answer the following question using ONLY the provided context. 
If the answer is not contained in the context, say "I don't know based on the provided evidence".
Always be supportive, concise, and professional.

Context:
{context}

Question:
{question}
"""
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content

# --- UI Setup ---
st.title("🏃‍♀️ Athletica")
st.markdown("**Evidence-based Nutrition & Physiology Assistant for Female Athletes**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Athletica. Ask me any question about female athlete physiology, nutrition, the menstrual cycle, or RED-S."}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If this assistant message has associated citations, display them
        if "citations" in message:
            with st.expander("📚 Ver Evidencia Médica Recuperada"):
                for idx, citation in enumerate(message["citations"], 1):
                    st.markdown(f"**Chunk {idx}:**\n{citation['content']}")
                    st.caption(f"📖 *{citation['title']}* - {citation['authors']} (PMID: {citation['pmid']})")
                    st.divider()

# React to user input
if prompt := st.chat_input("E.g., How does the menstrual cycle affect my training?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la literatura científica..."):
            # 1. Retrieve
            retrieved_ids = search_knn(prompt, top_k=5)
            
            # 2. Build Context
            citations = [chunk_dict[cid] for cid in retrieved_ids]
            context_text = "\n\n---\n\n".join([c["content"] for c in citations])
        
        with st.spinner("Analizando evidencia..."):
            # 3. Generate Answer
            answer = generate_answer(prompt, context_text)
            
            # Display answer
            st.markdown(answer)
            
            # Display expander with citations
            with st.expander("📚 Ver Evidencia Médica Recuperada"):
                for idx, citation in enumerate(citations, 1):
                    st.markdown(f"**Chunk {idx}:**\n{citation['content']}")
                    st.caption(f"📖 *{citation['title']}* - {citation['authors']} (PMID: {citation['pmid']})")
                    st.divider()

    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "citations": citations
    })

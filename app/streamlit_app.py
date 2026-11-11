"""Streamlit UI for the RAG assistant.

A thin front end over the FastAPI backend: type a question, get an answer
with source citations. Start the API first (``uvicorn app.main:app``), then
run this with ``streamlit run app/streamlit_app.py``.
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="RAG Assistant (Bedrock)", page_icon="📚")
st.title("📚 RAG Assistant")
st.write(
    "Ask a question about the documents in the `data/` folder. "
    "Answers cite the sources they came from."
)

# --- Sidebar: rebuild the index ------------------------------------------------
with st.sidebar:
    st.header("Documents")
    st.write("Rebuild the search index from the files in `data/`.")
    if st.button("🔄 Rebuild index"):
        with st.spinner("Indexing documents..."):
            try:
                response = requests.post(f"{API_URL}/ingest", timeout=300)
                response.raise_for_status()
                data = response.json()
                st.success(
                    f"Indexed {data['chunks_indexed']} chunks from "
                    f"{data['documents_loaded']} documents."
                )
            except requests.RequestException as exc:
                st.error(f"Could not reach the API at {API_URL}: {exc}")
    st.caption(f"API: {API_URL}")

# --- Main: ask a question --------------------------------------------------------
question = st.text_input(
    "Your question",
    placeholder="e.g. How many days of paid time off do employees get?",
)
top_k = st.slider("Sources to retrieve", min_value=1, max_value=10, value=4)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching documents and asking Bedrock..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": top_k},
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Could not reach the API at {API_URL}: {exc}")
            st.stop()

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    if result["sources"]:
        for source in result["sources"]:
            page = f", page {source['page']}" if source.get("page") else ""
            st.write(f"- {source['source']}{page}")
    else:
        st.write("No sources were retrieved for this question.")

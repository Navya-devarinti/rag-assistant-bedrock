"""FastAPI backend for the RAG assistant.

Endpoints
---------
- ``GET  /health`` -- liveness check (does not need AWS).
- ``POST /ingest`` -- rebuild the FAISS index from the files in ``data/``.
- ``POST /query``  -- ask a question, get an answer with source citations.

The Bedrock clients and the vector store are loaded lazily on first use, so
importing this module (and starting the server) does not need AWS
credentials. The first ``/query`` or ``/ingest`` call is the one that talks
to Bedrock.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.chain import answer_question, get_llm
from rag.config import get_settings
from rag.embeddings import get_embeddings
from rag.ingest import build_index, chunk_documents, load_documents, load_index
from rag.retriever import get_retriever

app = FastAPI(
    title="RAG Assistant (AWS Bedrock)",
    description="Ask questions about your documents; answers cite their sources.",
    version="1.0.0",
)

# Lazy singletons: created on first use, never at import time.
_state: dict = {"vector_store": None, "llm": None}


class QueryRequest(BaseModel):
    """Body for POST /query."""

    question: str = Field(..., min_length=1, description="The question to ask.")
    top_k: int = Field(default=4, ge=1, le=10, description="Chunks to retrieve.")


class QueryResponse(BaseModel):
    """Response for POST /query."""

    answer: str
    sources: list[dict]
    question: str


class IngestResponse(BaseModel):
    """Response for POST /ingest."""

    documents_loaded: int
    chunks_indexed: int
    index_dir: str


def _get_vector_store():
    """Load the FAISS index from disk (cached after the first load)."""
    if _state["vector_store"] is None:
        settings = get_settings()
        if not settings.index_dir.is_dir():
            raise HTTPException(
                status_code=503,
                detail=(
                    "No FAISS index found. Run POST /ingest first, or "
                    "`python scripts/build_index.py`."
                ),
            )
        _state["vector_store"] = load_index(settings.index_dir, get_embeddings(settings))
    return _state["vector_store"]


def _get_llm():
    """Return the Bedrock chat client (cached after the first call)."""
    if _state["llm"] is None:
        _state["llm"] = get_llm()
    return _state["llm"]


@app.get("/health")
def health() -> dict:
    """Liveness check. Always available, even without AWS credentials."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    """Rebuild the FAISS index from the files in the data folder."""
    settings = get_settings()
    documents = load_documents(settings.data_dir)
    if not documents:
        raise HTTPException(
            status_code=400,
            detail=f"No supported documents found in {settings.data_dir}.",
        )
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    _state["vector_store"] = build_index(chunks, get_embeddings(settings), settings.index_dir)
    return IngestResponse(
        documents_loaded=len(documents),
        chunks_indexed=len(chunks),
        index_dir=str(settings.index_dir),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question from the indexed documents, with citations."""
    settings = get_settings()
    retriever = get_retriever(_get_vector_store(), k=request.top_k)
    result = answer_question(request.question, retriever, _get_llm())
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        question=request.question,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

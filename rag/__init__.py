"""RAG assistant package.

The RAG logic lives here, split into small modules:

- ``config``     -- settings from environment variables / .env
- ``embeddings`` -- Bedrock embeddings client (created lazily)
- ``ingest``     -- load documents from data/, chunk them, build a FAISS index
- ``retriever``  -- retrieval helpers and citation formatting
- ``chain``      -- the question-answering chain (retrieve -> prompt -> LLM)

Nothing in this package creates an AWS client at import time, so importing
it does not require AWS credentials.
"""

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings"]

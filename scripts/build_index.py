"""Command-line helper: build the FAISS index from the data folder.

Usage:
    python scripts/build_index.py

This loads every supported document from ``data/``, chunks it, embeds the
chunks with Amazon Titan Text Embeddings V2 on Bedrock, and saves a FAISS
index to ``faiss_index/``. Needs AWS credentials with Bedrock access
(see README.md). The FastAPI ``POST /ingest`` endpoint does the same thing.
"""

import sys
from pathlib import Path

# Allow running as `python scripts/build_index.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import get_settings
from rag.embeddings import get_embeddings
from rag.ingest import build_index, chunk_documents, load_documents


def main() -> None:
    settings = get_settings()
    documents = load_documents(settings.data_dir)
    if not documents:
        print(f"No supported documents found in {settings.data_dir}.")
        raise SystemExit(1)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    build_index(chunks, get_embeddings(settings), settings.index_dir)
    print(
        f"Indexed {len(chunks)} chunks from {len(documents)} documents "
        f"-> {settings.index_dir}"
    )


if __name__ == "__main__":
    main()

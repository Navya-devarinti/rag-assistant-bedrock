"""Document loading, chunking, and FAISS index building.

Supported files in the data folder: ``.pdf``, ``.txt`` and ``.md``.
Each loaded document keeps its file name in ``metadata["source"]`` so answers
can cite where they came from.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def _load_one_file(path: Path) -> list[Document]:
    """Load a single file into LangChain documents."""
    if path.suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    else:  # .txt and .md are plain text
        loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()
    # Normalise the source metadata to just the file name for clean citations.
    for doc in documents:
        doc.metadata["source"] = path.name
    return documents


def load_documents(data_dir: str | Path) -> list[Document]:
    """Load every supported document from ``data_dir`` (sorted by name).

    Files with unsupported extensions are skipped with a short warning.
    Returns an empty list when the folder has no supported files.
    """
    data_dir = Path(data_dir)
    documents: list[Document] = []
    if not data_dir.is_dir():
        print(f"Warning: data directory not found: {data_dir}")
        return documents
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            print(f"Skipping unsupported file: {path.name}")
            continue
        documents.extend(_load_one_file(path))
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into overlapping chunks for embedding.

    Overlap keeps sentences that straddle a chunk boundary in context.
    Metadata (including the source file) is copied onto every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_index(
    documents: list[Document],
    embeddings,
    index_dir: str | Path,
) -> FAISS:
    """Embed chunked documents and save a FAISS index to ``index_dir``."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(str(index_dir))
    return vector_store


def load_index(index_dir: str | Path, embeddings) -> FAISS:
    """Load a previously saved FAISS index from ``index_dir``.

    ``allow_dangerous_deserialization`` is required because FAISS stores its
    document metadata as a pickle file. Only load indexes you built yourself.
    """
    return FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )

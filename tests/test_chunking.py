"""Tests for document chunking (rag.ingest.chunk_documents)."""

from langchain_core.documents import Document

from rag.ingest import chunk_documents

LONG_TEXT = (
    "Bluepeak Analytics is a remote-first company. Employees may work from "
    "any location in the United States. Teams hold a weekly video standup "
    "on Mondays at 10:00 a.m. Eastern, and cameras are optional. "
) * 12  # ~2,000 characters


def test_long_document_is_split_into_multiple_chunks():
    docs = [Document(page_content=LONG_TEXT, metadata={"source": "handbook.md"})]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1


def test_chunks_respect_chunk_size():
    docs = [Document(page_content=LONG_TEXT, metadata={"source": "handbook.md"})]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    for chunk in chunks:
        assert len(chunk.page_content) <= 500


def test_chunk_metadata_preserves_source():
    docs = [Document(page_content=LONG_TEXT, metadata={"source": "handbook.md"})]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert chunks
    for chunk in chunks:
        assert chunk.metadata["source"] == "handbook.md"


def test_all_original_text_survives_chunking():
    docs = [Document(page_content=LONG_TEXT, metadata={"source": "handbook.md"})]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    joined = "".join(chunk.page_content for chunk in chunks)
    # Every sentence of the original should appear in at least one chunk.
    assert "remote-first company" in joined
    assert "weekly video standup" in joined


def test_empty_input_returns_empty_list():
    assert chunk_documents([]) == []


def test_short_document_is_not_split():
    docs = [Document(page_content="Short text.", metadata={"source": "faq.md"})]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."

"""Tests for retrieval formatting and the FAISS index round trip.

No real AWS calls are made: the FAISS tests use a deterministic fake
embeddings class instead of Bedrock.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.ingest import build_index, load_index
from rag.retriever import format_documents, format_sources, get_retriever


class FakeEmbeddings(Embeddings):
    """Deterministic stand-in for Bedrock embeddings (no network calls)."""

    dim = 8

    def _vector_for(self, text: str) -> list[float]:
        # A stable vector derived from the text length, so similar-length
        # texts end up near each other. Good enough for index round trips.
        base = float(len(text) % 10) + 0.1
        return [base + i * 0.01 for i in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)


def _sample_docs() -> list[Document]:
    return [
        Document(
            page_content="Employees get 20 days of paid time off per year.",
            metadata={"source": "handbook.md"},
        ),
        Document(
            page_content="Raw event logs are kept for 90 days.",
            metadata={"source": "faq.md", "page": 0},
        ),
        Document(
            page_content="Returns are free within 30 days of delivery.",
            metadata={"source": "returns-policy.txt"},
        ),
    ]


def test_format_documents_numbers_citations_in_order():
    formatted = format_documents(_sample_docs())
    assert "[1] (source: handbook.md)" in formatted
    assert "[2] (source: faq.md, page 1)" in formatted
    assert "[3] (source: returns-policy.txt)" in formatted
    assert "20 days of paid time off" in formatted
    assert "90 days" in formatted


def test_format_documents_empty_input_returns_empty_string():
    assert format_documents([]) == ""


def test_format_sources_deduplicates_same_file_and_page():
    docs = [
        Document(page_content="a", metadata={"source": "handbook.md"}),
        Document(page_content="b", metadata={"source": "handbook.md"}),
        Document(page_content="c", metadata={"source": "faq.md", "page": 2}),
    ]
    sources = format_sources(docs)
    assert sources == [
        {"source": "handbook.md", "page": None},
        {"source": "faq.md", "page": 3},  # pages are reported 1-based
    ]


def test_format_sources_empty_input_returns_empty_list():
    assert format_sources([]) == []


def test_faiss_index_round_trip(tmp_path):
    index_dir = tmp_path / "faiss_index"
    store = build_index(_sample_docs(), FakeEmbeddings(), index_dir)
    assert (index_dir / "index.faiss").exists()

    loaded = load_index(index_dir, FakeEmbeddings())
    results = loaded.similarity_search("paid time off", k=2)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_retriever_respects_top_k(tmp_path):
    store = build_index(_sample_docs(), FakeEmbeddings(), tmp_path / "idx")
    retriever = get_retriever(store, k=2)
    results = retriever.invoke("how long are logs kept?")
    assert len(results) == 2

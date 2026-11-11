"""Retrieval helpers: turn retrieved chunks into cited context.

Citations are the numbered markers ``[1]``, ``[2]``, ... that the language
model is asked to use in its answer. Each marker maps to one source chunk,
so a reader can check where every claim came from.
"""

from langchain_core.documents import Document


def format_documents(documents: list[Document]) -> str:
    """Format retrieved chunks as numbered context for the LLM prompt.

    Example output::

        [1] (source: handbook.md)
        <chunk text>
        ---
        [2] (source: faq.md)
        <chunk text>

    Returns an empty string when no documents were retrieved.
    """
    if not documents:
        return ""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"(source: {source}" + (f", page {page + 1}" if page is not None else "") + ")"
        blocks.append(f"[{i}] {label}\n{doc.page_content.strip()}")
    return "\n---\n".join(blocks)


def format_sources(documents: list[Document]) -> list[dict]:
    """Return a de-duplicated list of sources for the retrieved chunks.

    Each entry is ``{"source": <file name>, "page": <1-based page or None>}``.
    """
    seen: set[tuple] = set()
    sources: list[dict] = []
    for doc in documents:
        page = doc.metadata.get("page")
        key = (doc.metadata.get("source", "unknown"), page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {"source": key[0], "page": (page + 1) if page is not None else None}
        )
    return sources


def get_retriever(vector_store, k: int = 4):
    """Return a similarity-search retriever over a FAISS vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})

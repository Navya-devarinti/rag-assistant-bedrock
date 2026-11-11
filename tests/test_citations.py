"""Tests for citation handling in the question-answering chain.

The retriever and the LLM are fakes, so no Bedrock or boto3 calls happen.
"""

from langchain_core.documents import Document

from rag.chain import answer_question


class FakeRetriever:
    """Returns a fixed set of documents regardless of the question."""

    def __init__(self, documents):
        self.documents = documents

    def invoke(self, question: str):
        return self.documents


class FakeLLM:
    """Records the prompt it receives and returns a canned answer."""

    def __init__(self, answer: str):
        self.answer = answer
        self.last_prompt = None

    def invoke(self, prompt: str):
        self.last_prompt = prompt

        class _Message:
            def __init__(self, content):
                self.content = content

        return _Message(self.answer)


def _sample_docs():
    return [
        Document(
            page_content="Employees get 20 days of paid time off per year.",
            metadata={"source": "handbook.md"},
        ),
        Document(
            page_content="There are 10 company holidays per year.",
            metadata={"source": "handbook.md", "page": 0},
        ),
    ]


def test_answer_returns_citations_and_sources():
    retriever = FakeRetriever(_sample_docs())
    llm = FakeLLM("Employees get 20 days of PTO [1] and 10 holidays [2].")
    result = answer_question("How much time off do employees get?", retriever, llm)

    assert result["answer"] == "Employees get 20 days of PTO [1] and 10 holidays [2]."
    assert result["sources"] == [
        {"source": "handbook.md", "page": None},
        {"source": "handbook.md", "page": 1},
    ]


def test_prompt_contains_numbered_context_with_sources():
    retriever = FakeRetriever(_sample_docs())
    llm = FakeLLM("An answer.")
    answer_question("How much time off?", retriever, llm)

    assert "[1] (source: handbook.md)" in llm.last_prompt
    assert "[2] (source: handbook.md, page 1)" in llm.last_prompt
    assert "20 days of paid time off" in llm.last_prompt
    assert "How much time off?" in llm.last_prompt


def test_no_documents_returns_empty_sources_and_honest_context():
    retriever = FakeRetriever([])
    llm = FakeLLM("I don't know based on the provided documents.")
    result = answer_question("What is the meaning of life?", retriever, llm)

    assert result["sources"] == []
    assert "No relevant documents were found." in llm.last_prompt
    assert result["answer"].startswith("I don't know")


def test_answer_keeps_citation_markers_verbatim():
    retriever = FakeRetriever(_sample_docs())
    llm = FakeLLM("See [2] for holidays and [1] for PTO.")
    result = answer_question("Tell me about time off.", retriever, llm)
    assert "[2]" in result["answer"] and "[1]" in result["answer"]

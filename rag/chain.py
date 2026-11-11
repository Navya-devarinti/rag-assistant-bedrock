"""Question-answering chain: retrieve -> build prompt -> call Bedrock LLM.

The flow is deliberately simple and explicit (no hidden chain machinery):

1. Retrieve the top-k most similar chunks for the question.
2. Format them as numbered context, e.g. ``[1] (source: handbook.md) ...``.
3. Ask Claude 3.5 Sonnet on Bedrock to answer *only* from that context,
   citing sources as ``[1]``, ``[2]``, ...
4. Return the answer together with the list of sources.

The Bedrock LLM client is only built when :func:`get_llm` is called, so
importing this module does not require AWS credentials.
"""

from langchain_aws import ChatBedrock

from .config import Settings, get_settings
from .retriever import format_documents, format_sources

# The model must answer from the retrieved context only, and must say when
# the context does not contain the answer, instead of guessing.
QA_PROMPT = """You are a helpful assistant that answers questions using ONLY the
context below. Every factual claim in your answer must cite the source it
came from, using the numbered markers like [1], [2] that appear in the
context. If the context does not contain the information needed to answer
the question, say so clearly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


def get_llm(settings: Settings | None = None) -> ChatBedrock:
    """Return a Bedrock chat client for Claude 3.5 Sonnet.

    Parameters
    ----------
    settings:
        Optional settings object. When omitted, settings are read from the
        environment (see :mod:`rag.config`).
    """
    settings = settings or get_settings()
    return ChatBedrock(
        model_id=settings.llm_model_id,
        region_name=settings.aws_region,
        model_kwargs={"temperature": settings.temperature},
    )


def answer_question(question: str, retriever, llm) -> dict:
    """Answer one question and return the answer plus its sources.

    Parameters
    ----------
    question:
        The user's question.
    retriever:
        A LangChain retriever (see :func:`rag.retriever.get_retriever`).
    llm:
        A chat model with an ``invoke(prompt)`` method returning a message
        with a ``content`` attribute (e.g. ``ChatBedrock``).

    Returns
    -------
    dict with keys:
        ``answer``  -- the model's answer text (with ``[n]`` citations),
        ``sources`` -- de-duplicated source list from :func:`format_sources`,
        ``context`` -- the formatted context that was sent to the model.
    """
    documents = retriever.invoke(question)
    context = format_documents(documents)
    if not context:
        context = "No relevant documents were found."
    prompt = QA_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return {
        "answer": answer,
        "sources": format_sources(documents),
        "context": context,
    }

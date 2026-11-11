"""Bedrock embeddings, created lazily.

The Bedrock client is only built when :func:`get_embeddings` is called, so
importing this module (or the FastAPI app) does not require AWS credentials.
"""

from langchain_aws import BedrockEmbeddings

from .config import Settings, get_settings


def get_embeddings(settings: Settings | None = None) -> BedrockEmbeddings:
    """Return a Bedrock embeddings client for Amazon Titan Text Embeddings V2.

    Parameters
    ----------
    settings:
        Optional settings object. When omitted, settings are read from the
        environment (see :mod:`rag.config`).
    """
    settings = settings or get_settings()
    return BedrockEmbeddings(
        model_id=settings.embedding_model_id,
        region_name=settings.aws_region,
    )

"""Tests for configuration loading and lazy AWS client creation."""

import os

from rag.config import get_settings


def test_defaults_without_env_vars(monkeypatch):
    for key in (
        "AWS_REGION",
        "BEDROCK_EMBEDDING_MODEL_ID",
        "BEDROCK_LLM_MODEL_ID",
        "RAG_CHUNK_SIZE",
        "RAG_CHUNK_OVERLAP",
        "RAG_TOP_K",
        "RAG_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = get_settings()
    assert settings.aws_region == "us-east-1"
    assert settings.embedding_model_id == "amazon.titan-embed-text-v2:0"
    assert settings.llm_model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.top_k == 4
    assert settings.temperature == 0.0


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("BEDROCK_LLM_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_TEMPERATURE", "0.3")
    settings = get_settings()
    assert settings.aws_region == "eu-west-1"
    assert settings.top_k == 7
    assert settings.temperature == 0.3


def test_no_aws_clients_created_at_import(monkeypatch):
    """Importing the app must not touch AWS, even with no credentials set."""
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    import app.main as app_main

    # Lazy singletons stay empty until the first /query or /ingest call.
    assert app_main._state == {"vector_store": None, "llm": None}
    assert app_main.app.title == "RAG Assistant (AWS Bedrock)"

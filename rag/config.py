"""Configuration for the RAG assistant.

All settings come from environment variables, loaded from a local ``.env``
file when one exists (see ``.env.example``). Nothing in this module imports
boto3 or creates an AWS client, so importing it never needs AWS credentials.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if one exists.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """All tunable settings for the RAG pipeline."""

    # AWS region where Bedrock is called, e.g. "us-east-1".
    aws_region: str = "us-east-1"

    # Bedrock model IDs. These are configurable because Bedrock model IDs
    # change over time and availability varies by region.
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    llm_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    # Where source documents live and where the FAISS index is stored.
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "faiss_index")

    # Chunking: how documents are split before embedding.
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval: how many chunks to pull in as context for each question.
    top_k: int = 4

    # LLM sampling temperature (0 = deterministic, good for factual Q&A).
    temperature: float = 0.0


def get_settings() -> Settings:
    """Build a Settings object, honouring environment variables when set."""

    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except ValueError:
            return default

    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except ValueError:
            return default

    return Settings(
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        embedding_model_id=os.environ.get(
            "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
        ),
        llm_model_id=os.environ.get(
            "BEDROCK_LLM_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
        ),
        data_dir=Path(os.environ.get("RAG_DATA_DIR", PROJECT_ROOT / "data")),
        index_dir=Path(os.environ.get("RAG_INDEX_DIR", PROJECT_ROOT / "faiss_index")),
        chunk_size=_env_int("RAG_CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 200),
        top_k=_env_int("RAG_TOP_K", 4),
        temperature=_env_float("RAG_TEMPERATURE", 0.0),
    )

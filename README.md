# RAG Assistant on AWS Bedrock

Ask questions about your own documents and get answers **with source
citations** — built with Python, LangChain, Amazon Bedrock, and FAISS.

This is a learning project: the code is small, commented, and written to be
read. It shows the full retrieval-augmented generation (RAG) loop that
AI/ML engineer roles ask about: ingest → embed → retrieve → generate.

## What it does

1. **Ingest** — reads PDF, `.txt`, and `.md` files from `data/`, splits them
   into overlapping chunks.
2. **Embed** — turns each chunk into a vector with **Amazon Titan Text
   Embeddings V2** on Bedrock.
3. **Retrieve** — stores the vectors in a local **FAISS** index and, for each
   question, pulls back the most similar chunks.
4. **Generate** — sends the question plus the retrieved chunks to
   **Claude 3.5 Sonnet** on Bedrock, which answers using *only* that
   context and cites its sources as `[1]`, `[2]`, ...

## Architecture

```
                  ┌──────────────────────┐
  data/*.pdf      │  ingest.py           │   chunk + embed
  data/*.txt ───▶ │  load → chunk →      │ ────────────────┐
  data/*.md       │  Titan Embeddings V2 │                 ▼
                  └──────────────────────┘        ┌─────────────┐
                                                  │ FAISS index │
                                                  │ (local)     │
  ┌──────────────┐                                └──────┬──────┘
  │ Streamlit UI │   question                            │ top-k chunks
  │ (ask + view  │ ─────────────┐                        ▼
  │  sources)    │              ▼              ┌───────────────────┐
  └──────┬───────┘     ┌────────────────┐     │ chain.py          │
         │            │ FastAPI        │────▶│ retrieve → prompt │
         └───────────▶│ /query /ingest │     │ → Claude 3.5      │
                       └────────────────┘     │   Sonnet (Bedrock)│
                                              └───────────────────┘
```

Everything in `rag/` is the reusable RAG logic. The FastAPI backend
(`app/main.py`) and the Streamlit UI (`app/streamlit_app.py`) are thin
layers over it.

## Project structure

```
rag-assistant-bedrock/
├── rag/                    # RAG logic (the interesting part)
│   ├── config.py           # settings from .env / environment
│   ├── embeddings.py       # Bedrock Titan embeddings (lazy client)
│   ├── ingest.py           # load docs, chunk, build/load FAISS index
│   ├── retriever.py        # retrieval + citation formatting
│   └── chain.py            # retrieve → prompt → Claude 3.5 Sonnet
├── app/
│   ├── main.py             # FastAPI: /health, /ingest, /query
│   └── streamlit_app.py    # simple chat UI
├── scripts/
│   └── build_index.py      # CLI: build the FAISS index once
├── data/                   # sample documents (fictional, written for this demo)
│   ├── employee-handbook.md
│   ├── faq.md
│   ├── returns-policy.txt
│   └── getting-started.pdf
├── tests/                  # pytest suite (mocked Bedrock, no AWS calls)
├── .env.example            # copy to .env and fill in
├── requirements.txt
└── README.md
```

## Quick start

Requires Python 3.11+ and an AWS account with Bedrock access (see below).

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure AWS (copy the example and edit it)
cp .env.example .env
# Set AWS_REGION and, if you use access keys, AWS_ACCESS_KEY_ID and
# AWS_SECRET_ACCESS_KEY. The Bedrock model IDs already have defaults.

# 3. Build the search index from the sample documents in data/
python scripts/build_index.py

# 4. Start the API (in one terminal)
uvicorn app.main:app --reload

# 5. Start the UI (in another terminal)
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501, ask e.g. *"How many days of paid time
off do employees get?"* and you should get an answer citing
`employee-handbook.md`.

You can also call the API directly:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How long are raw event logs kept?"}'
```

To add your own documents, drop files into `data/` and rebuild the index
(`POST /ingest`, the sidebar button, or `python scripts/build_index.py`).

## Pointing it at real AWS

1. **Enable the models in Bedrock.** In the AWS console, go to Bedrock →
   Model access, and request access to *Amazon Titan Text Embeddings V2*
   and *Anthropic Claude 3.5 Sonnet*. Access is usually granted quickly.
2. **Credentials.** The app uses boto3's standard chain: environment
   variables, `~/.aws/credentials`, or an IAM role. No credentials are ever
   hardcoded — clients are created lazily, so the app imports and starts
   even without credentials; only the first Bedrock call needs them.
3. **Model IDs.** If a model ID is unavailable in your region, change
   `BEDROCK_EMBEDDING_MODEL_ID` / `BEDROCK_LLM_MODEL_ID` in `.env`. The
   current defaults are `amazon.titan-embed-text-v2:0` and
   `anthropic.claude-3-5-sonnet-20241022-v2:0`.
4. **Region.** Bedrock model availability varies by region; `us-east-1` is
   the safest default.

## Production notes

- **Vector store:** FAISS here is local and free — great for a demo and for
  learning. For production, swap it for **Amazon OpenSearch Serverless**
  (managed vector search) or another managed store; only `rag/ingest.py`
  and `rag/retriever.py` would change.
- **Chunking:** `chunk_size` / `chunk_overlap` in `.env` control how
  documents are split. Smaller chunks retrieve more precisely; larger
  chunks keep more context.
- **Guardrails:** the prompt tells the model to answer only from retrieved
  context and to say when it does not know. For production, add Bedrock
  Guardrails and a human review step for sensitive answers.

## Cost notes

- **FAISS is free** — it runs locally, no managed service involved.
- **Bedrock is pay-per-use.** You pay per token for embeddings and for
  Claude input/output tokens; there is no hourly charge for the models
  themselves. A demo workload (a few dozen documents, a few hundred
  questions) typically costs a few dollars at most — but check the current
  [AWS Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) before
  running large jobs, and set a billing alarm.
- The Streamlit UI and FastAPI server cost nothing beyond the machine they
  run on.

## Tests

```bash
pytest -v
```

The suite uses fake embeddings, a fake retriever, and a fake LLM, so it
runs with **no AWS credentials and no network calls**. It covers chunking,
retrieval formatting, citation handling, config loading, and the lazy
client creation that lets the app import without credentials.

## License

MIT — see [LICENSE](LICENSE).

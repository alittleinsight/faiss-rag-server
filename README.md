# Local RAG Server – Private, GPU-Accelerated  
**Your personal "ChatGPT for your files" – 100% local, no data leaves your machine**

![RAG Server](https://img.shields.io/badge/RAG-Local%20·%20Private%20·%20Fast-blue)  
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)  
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A fast, containerized Retrieval-Augmented Generation (RAG) server with full document management, hybrid search, and compatibility with **LM Studio, Cursor, Continue.dev, Open WebUI**, and any OpenAI client.

## Features

| Feature                                 | Status  | Details |
|----------------------------------------|--------|--------|
| GPU-accelerated embeddings & FAISS search | Done    | Uses your RTX 2060 |
| Automatic context injection             | Done    | No hallucinations — answers come from your docs |
| Full document CRUD                      | Done    | List, add, delete, incremental index updates |
| Add documents via web & API             | Done    | Index from source path (no file copy) |
| Persistent knowledge base               | Done    | `./documents` and `./vectorstore` survive container restarts |
| Swagger UI + simple dashboard           | Done    | http://localhost:8004/docs & http://localhost:8004 |
| Zero external dependencies              | Done    | No Pinecone, no OpenAI, no internet required |

## Quick Start (Docker – Recommended)

```bash
# 1. Clone or download this project
git clone https://github.com/yourname/local-rag-server.git
cd local-rag-server

# 2. Put your documents in the `documents/` folder
#    (PDFs, .txt, .md, code files – anything text-based)

# 3. Build & start (first time only)
docker compose up --build

# 4. Daily use (instant start)
docker compose up -d
```

**Open in browser:**
- Dashboard to add and delete documents: http://localhost:8004/dashboard
- Full API docs: http://localhost:8004/docs
- MCP: http://localhost:8004/mcp

That’s it — ask questions about your documents instantly!

## API Endpoints

| Method   | Endpoint                        | Description                                    |
|----------|----------------------------------|------------------------------------------------|
| `GET`    | `/dashboard`                     | Simple web dashboard for adding/deleting documents                             |
| `GET`    | `/docs`                          | Interactive Swagger UI                         |
| `GET`    | `/api/documents`                 | List indexed files with `filename` and `source_path` |
| `DELETE` | `/api/documents/{filename}`      | Remove matching chunks from index only (source file remains) |
| `DELETE` | `/api/document?source_path=...`  | Remove exactly one indexed source path from index only |
| `GET`    | `/api/search?q=your+question`    | Hybrid vector + keyword search                 |
| `POST`   | `/api/add`                       | Add a file to index from `source_path` (no copy into `documents/`) |
| `GET`    | `/mcp`                           | MCP server streaming endpoint (initial connection + heartbeat) |
| `POST`   | `/mcp`                           | MCP JSON-RPC dispatcher (`initialize`, `tools/list`, `tools/call`) |

### Example: Index a file directly from its original path (no copy)
```bash
curl -X POST "http://localhost:8004/api/add" \
  -F "source_path=/workspace/external-docs/policy.pdf"
```

The server verifies the file exists and is readable before indexing.
If the same normalized `source_path` with unchanged file content is already indexed, the request returns success with `"skipped": true` and does not duplicate chunks.
If file content changed for an existing `source_path`, the server performs an incremental in-place update for that file only.
When running in Docker, use container-visible paths such as `/workspace/documents/file.pdf` (or relative `documents/file.pdf`).

### Example: Search
```bash
curl "http://localhost:8004/api/search?q=remote%20work%20policy"
```

### Example: List Documents
```bash
curl -X GET "http://localhost:8004/api/documents"
```

Example response:

```json
{
  "documents": [
    {
      "filename": "policy.pdf",
      "source_path": "C:/.../documents/policy.pdf"
    }
  ],
  "count": 1
}
```

### Example: Delete a Document
```bash
curl -X DELETE "http://localhost:8004/api/documents/sample.pdf"
```

If the filename exists under multiple source paths, the legacy filename endpoint returns HTTP 409 and asks for source path delete.

Preferred exact delete:

```bash
curl -X DELETE "http://localhost:8004/api/document?source_path=/workspace/external-docs/sample.pdf"
```

Delete response now reflects index-only behavior:

```json
{
  "deleted": "sample.pdf",
  "chunks_removed": 6,
  "index_only_delete": true
}
```

### Example: MCP Tool Call
```bash
curl -X 'POST' \
  'http://localhost:8004/mcp' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
"method":"tools/call",
"params":{
  "name":"retrieve_context",
  "arguments":{
    "query":"SHELLS AND COMMAND-LINE TOOLS",
    "top_k":3
    }
  }
}'
```

### MCP Tools

- `retrieve_context` - retrieve top-k chunks for a query.
- `list_documents` - list indexed documents including `filename` and `source_path`.
- `add_document` - add a file from optional `source_path`, or rebuild from `./documents` if omitted.

Client parsing tip: MCP tool handlers return payloads as text blocks in `result.content[0].text`; parse that string as JSON to consume structured fields.

Example `add_document` call:

```bash
curl -X 'POST' \
  'http://localhost:8004/mcp' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
"method":"tools/call",
"params":{
  "name":"add_document",
  "arguments":{}
  }
}'
```

Example `add_document` call with `source_path`:

```bash
curl -X 'POST' \
  'http://localhost:8004/mcp' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
"method":"tools/call",
"params":{
  "name":"add_document",
  "arguments":{
    "source_path":"/workspace/external-docs/policy.pdf"
  }
  }
}'
```

Example `add_document` response:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"status\": \"success\",\n  \"indexed_documents\": 4,\n  \"indexed_chunks\": 52\n}"
      }
    ]
  }
}
```

Example `list_documents` response:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"documents\": [\n    {\n      \"filename\": \"policy.pdf\",\n      \"source_path\": \"C:/.../documents/policy.pdf\"\n    }\n  ],\n  \"count\": 1\n}"
      }
    ]
  }
}
```

## Check MCP operation with MCP Inspector

https://modelcontextprotocol.io/docs/tools/inspector#python
https://github.com/modelcontextprotocol/inspector

``` bash
docker run --rm \
  -p 127.0.0.1:6274:6274 \
  -p 127.0.0.1:6277:6277 \
  -e HOST=0.0.0.0 \
  -e MCP_AUTO_OPEN_ENABLED=false \
  ghcr.io/modelcontextprotocol/inspector:latest
```
When it starts, you’ll see:
A session token printed in the logs
(you must paste this into the UI unless you use the pre-filled URL)

A link like:
http://localhost:6274/?MCP_PROXY_AUTH_TOKEN=...

Open that URL and you’re in.
To connect to your RAG server:
Transport Type - Streamable HTTP
URL - http://host.docker.internal:8004/mcp

## Configuration

Set runtime options with environment variables (for Docker, use `docker-compose.yml` or a `.env` file):

```python
# GPU/runtime
RAG_USE_GPU=true
RAG_GPU_INDEX=
RAG_FAISS_INDEX_DEVICE=gpu  # gpu | cpu
RAG_FAISS_SEARCH_DEVICE=gpu # gpu | cpu (search runtime index; mutable source-of-truth remains CPU)
CUDA_VISIBLE_DEVICES=0
DEFAULT_FAISS_DIM=1024

# embeddings.py
EMBEDDING_MODEL_LARGE = "BAAI/bge-large-en-v1.5"

# index.py
DOCUMENTS_PATH = Path("./documents")
VECTORSTORE_PATH = Path("./vectorstore")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3
```

If one GPU is unsupported by the current PyTorch build, the server now auto-skips it and falls back to another usable GPU (or CPU) instead of crashing.
Set `RAG_FAISS_INDEX_DEVICE=cpu` to keep the in-memory FAISS index on CPU (lower persistent VRAM use), or `gpu` to keep it on GPU (faster query/index ops on larger GPUs).
Set `RAG_FAISS_SEARCH_DEVICE=gpu` to accelerate read/search path with a GPU runtime index while preserving CPU mutable indexing for add/update/delete correctness.

Search runtime behavior:
- `RAG_FAISS_SEARCH_DEVICE=cpu`: both mutable indexing and query search run on CPU.
- `RAG_FAISS_SEARCH_DEVICE=gpu`: mutable index remains CPU source-of-truth, while query search attempts to run on a FAISS GPU runtime index.
- if FAISS GPU APIs or usable GPU are unavailable, search falls back to CPU automatically.

**Want bigger context or different model?**  
Just change `EMBEDDING_MODEL` to any Hugging Face sentence transformer (currently running, `BAAI/bge-large-en-v1.5` for better accuracy).

## Benchmarking (Real Corpus)

Run the real-corpus benchmark script against files in `./documents`:

```bash
# Run both CPU and GPU profiles (default)
docker compose run --rm --entrypoint python faiss-rag-server scripts/benchmark_real_corpus.py --profile both

# Run CPU-only (embeddings + FAISS on CPU)
docker compose run --rm --entrypoint python faiss-rag-server scripts/benchmark_real_corpus.py --profile cpu

# Run GPU-only (embeddings on cuda:0 + FAISS GPU indexing)
docker compose run --rm --entrypoint python faiss-rag-server scripts/benchmark_real_corpus.py --profile gpu
```

Notes:
- `--profile gpu` fails fast if FAISS GPU indexing is unavailable (it will not silently fall back to CPU).
- `--profile both` prints CPU and GPU timings plus speedup metrics.

## Testing

Run the incremental indexing test suite (Docker-first):

```bash
docker compose run --rm --entrypoint python faiss-rag-server -m unittest discover -s tests -v
```

Optional local (non-container) run:

```bash
python -m unittest discover -s tests -v
```

What this suite validates:
- incremental add/noop/update/delete flow by `source_path`
- filename-delete ambiguity when duplicate basenames exist in different folders
- catalog CRUD roundtrip for document/chunk metadata
- dashboard route serves extracted HTML template
- dashboard uses exact delete endpoint: `/api/document?source_path=...`

Note:
- tests use lightweight in-test fakes for FAISS/loaders/embeddings so they can run in environments without GPU or FAISS native binaries
- current Docker run status: `Ran 5 tests ... OK`

## Project Structure

```
faiss-rag-server/
├── app.py                  # Main FastAPI server (entry point)
├── api/                    # API endpoints for document and search operations
│   ├── api_endpoints.py    # Defines routes for add, search, delete, etc.
│   └── __init__.py         # Package initialization
├── mcp/                    # MCP dispatcher and tool definitions
│   ├── dispatcher.py       # Handles MCP tool calls
│   ├── tool_definitions.py # Definitions for MCP tools
│   ├── tool_registry.py    # Tool registration logic
│   └── __init__.py         # Package initialization
├── rag/                    # Retrieval-Augmented Generation logic
│   ├── docHandlers.py      # Document handling utilities
│   ├── embeddings.py       # Embedding generation logic
│   ├── index.py            # Index management (FAISS)
│   └── __init__.py         # Package initialization
├── documents/              # Optional local ingestion source directory
├── presentations/          # Slide decks and markdown outlines (kept out of RAG ingestion)
├── scripts/                # Utility scripts (e.g., deck generator)
├── vectorstore/            # Persistent FAISS index storage
├── Dockerfile              # Docker image configuration
├── docker-compose.yml      # Docker Compose setup
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore rules
```

## Presentation Deck Workflow

Deck source files are intentionally outside `./documents` so they are not embedded into the RAG index.
Source of truth for slide content is: presentations/faiss-rag-server-collab-deck.md
Update that markdown first, then mirror wording/structure here.

```bash
# Regenerate the collaborative deck
python scripts/generate_collab_deck.py

# Output files
presentations/faiss-rag-server-collab-deck.pptx
presentations/faiss-rag-server-collab-deck.md
```

## Troubleshooting

| Issue                                 | Fix |
|---------------------------------------|-----|
| Container crashes on startup          | Make sure `python-multipart` is in `requirements.txt` |
| sm_120 warning (RTX 5060 Ti)          | Harmless — your RTX 2060 is doing all the work |
| Slow indexing                         | Normal on first run — subsequent restarts are instant |

## Future (Dec 2025+)

- Full RTX 5060 Ti (Blackwell) support with PyTorch 2.7
- Variable chunk and overlap sizing based on doc type
- Include a re-ranker
- Include a sumarizer
- expand the embeddings to go multimodal (audio and image)


Enjoy!
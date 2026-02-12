# Local RAG Server – Private, GPU-Accelerated, OpenAI-Compatible  
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
| Full document CRUD                      | Done    | List, upload, delete, auto-reindex |
| File upload via web & API               | Done    | Drag & drop PDFs, .txt, .md, code files |
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
- Dashboard to upload and delete documents: http://localhost:8004/dashboard
- Full API docs: http://localhost:8004/docs
- MCP: http://localhost:8004/mcp

That’s it — ask questions about your documents instantly!

## API Endpoints

| Method   | Endpoint                        | Description                                    |
|----------|----------------------------------|------------------------------------------------|
| `GET`    | `/dashboard`                     | Simple web dashboard for uploading/deleting documents                          |
| `GET`    | `/docs`                          | Interactive Swagger UI                         |
| `GET`    | `/api/documents`                 | List all indexed files                         |
| `DELETE` | `/api/documents/{filename}`      | Delete file + auto-rebuild index               |
| `GET`    | `/api/search?q=your+question`    | Hybrid vector + keyword search                 |
| `POST`   | `/api/upload`                    | Upload file → auto-indexed                     |
| `GET`    | `/mcp`                           | MCP RAG server streaming endpoint/initial connection and heartbeat|
| `POST`   | `/mcp`                           | MCP RAG server dispatcher                      |
| `POST`   | `/mcp/call`                      | Call MCP tools like `retrieve_context`         |

### Example: Upload a file via curl
```bash
curl -X POST "http://localhost:8004/api/upload" \
  -F "file=@./documents/policy.pdf"
```

### Example: Search
```bash
curl "http://localhost:8004/api/search?q=remote%20work%20policy"
```

### Example: List Documents
```bash
curl -X GET "http://localhost:8004/api/documents"
```

### Example: Delete a Document
```bash
curl -X DELETE "http://localhost:8004/api/documents/sample.pdf"
```

### Example: MCP Tool Call
```bash
curl -X POST \
  'http://localhost:8004/mcp/call' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
   "tool": "retrieve_context",
   "params": {
     "query": "86xx memory card",
     "top_k": 10
   }
}'
```

## Check mcp operation with mcpinspector

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

## Configuration

All settings are in `config.py` (or create a `.env` file):

```python
# embeddings.py
EMBEDDING_MODEL_LARGE = "BAAI/bge-large-en-v1.5"
SUPPORTED_COMPUTE_CAPABILITY = (7, 5)  # example: sm_75
GPU_AVAILABLE = True

# index.py
DOCUMENTS_PATH = Path("./documents")
VECTORSTORE_PATH = Path("./vectorstore")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3
```

**Want bigger context or different model?**  
Just change `EMBEDDING_MODEL` to any Hugging Face sentence transformer (currently running, `BAAI/bge-large-en-v1.5` for better accuracy).

## Project Structure

```
faiss-rag-server/
├── app.py                  # Main FastAPI server (entry point)
├── api/                    # API endpoints for document and search operations
│   ├── api_endpoints.py    # Defines routes for upload, search, delete, etc.
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
├── documents/              # Directory for user-uploaded documents
├── vectorstore/            # Persistent FAISS index storage
├── Dockerfile              # Docker image configuration
├── docker-compose.yml      # Docker Compose setup
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore rules
```

## Troubleshooting

| Issue                                 | Fix |
|---------------------------------------|-----|
| Container crashes on startup          | Make sure `python-multipart` is in `requirements.txt` |
| sm_120 warning (RTX 5060 Ti)          | Harmless — your RTX 2060 is doing all the work |
| Slow indexing                         | Normal on first run — subsequent restarts are instant |

## Future (Dec 2025+)

- Full RTX 5060 Ti (Blackwell) support with PyTorch 2.7


Enjoy!
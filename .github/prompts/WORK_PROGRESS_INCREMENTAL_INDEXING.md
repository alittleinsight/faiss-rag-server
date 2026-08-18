# Incremental Indexing Work Progress

## Last Updated
- 2026-03-27

## Completed
- Added SQLite catalog module for source_path/chunk_id metadata in `rag/catalog.py`.
- Refactored index layer to mutable ID-based FAISS in `rag/index.py`.
- Implemented incremental file upsert (add/update/noop by file hash) in `rag/index.py`.
- Implemented delete by canonical source_path and compatibility delete-by-filename with ambiguity handling in `rag/index.py`.
- Refactored retrieval/list/add/delete handlers to use new index APIs in `rag/docHandlers.py`.
- Added preferred API delete endpoint by source path in `api/api_endpoints.py`.
- Updated README to reflect incremental behavior and source_path delete path.
- Added reproducible implementation prompt in `INCREMENTAL_INDEXING.prompt.md`.
- Added `tests/test_incremental_indexing.py` covering incremental add/noop/update/delete flow, filename ambiguity handling, and catalog CRUD.
- Added README testing instructions for running the incremental suite.
- Refactored dashboard HTML out of `app.py` into `templates/dashboard.html`.
- Updated dashboard delete flow to use exact-path endpoint: `/api/document?source_path=...`.
- Added `tests/test_dashboard.py` for dashboard route and endpoint wiring checks.
- Validated tests in Docker with: `docker compose run --rm --entrypoint python faiss-rag-server -m unittest discover -s tests -v` (5 passed).
- Added read-optimized hybrid search runtime mode in `rag/index.py`:
  - CPU mutable index remains source-of-truth for add/update/delete
  - optional GPU runtime search index enabled by `RAG_FAISS_SEARCH_DEVICE=gpu`
  - graceful fallback to CPU search when GPU path is unavailable
- Added `RAG_FAISS_SEARCH_DEVICE` to `docker-compose.yml` and documented behavior in `README.md`.

## In Progress
- Optional deeper runtime validation against real FAISS/native dependencies.

## Remaining
- Add or run explicit persistence-across-restart test against real FAISS/native runtime.
- Optionally add `/api/sync` for configured roots beyond `documents/`.
- Optionally expose delete-by-source-path MCP tool for non-HTTP clients.

## Notes for Resume
- Core mutation operations are now in `rag/index.py`:
  - `upsert_file_to_index(source_path)`
  - `delete_document_by_source_path(source_path)`
  - `delete_document_by_filename(filename)`
- Catalog DB path: `vectorstore/catalog.db`
- FAISS path: `vectorstore/index.faiss`
- Startup consistency check rebuilds FAISS from catalog text when `index.ntotal` mismatches catalog chunk count.

## Risks / Follow-ups
- Current startup mismatch recovery re-embeds catalog text; this is correct but can be slow for very large corpora.
- Mutable `IndexIDMap2(IndexFlatIP)` remains a correctness-first baseline; read-heavy performance is improved via optional GPU runtime search index, but IVF/compaction optimization work is still pending.

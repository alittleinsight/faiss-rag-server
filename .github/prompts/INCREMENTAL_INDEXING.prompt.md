# Prompt: Implement Incremental RAG Indexing (File-by-File)

You are implementing incremental embedding and FAISS indexing in this repository.

## Goal
Enable file-level add/update/delete without rebuilding the whole index:
- adding a file embeds/indexes only that file
- deleting a file removes only that file's vectors
- updating a file replaces vectors for that file only

Files may be anywhere on disk, not only `./documents`.

## Existing architecture
- FastAPI endpoints in `api/api_endpoints.py`
- RAG handlers in `rag/docHandlers.py`
- FAISS index logic in `rag/index.py`
- MCP tool handlers in `mcp/dispatcher.py`
- Dashboard route in `app.py` with HTML template in `templates/dashboard.html`

## Required implementation
1. Add persistent catalog metadata using SQLite (`vectorstore/catalog.db`):
   - documents table keyed by canonical source path
   - chunks table keyed by stable int64 chunk IDs
   - state table for monotonic next chunk id
2. Use `faiss.IndexIDMap2` and `add_with_ids` / `remove_ids` for mutable indexing.
3. Canonicalize source paths before dedupe/update/delete decisions.
4. Implement incremental operations:
   - upsert file by `source_path` with hash-based no-op detection
   - delete by `source_path`
   - compatibility delete by filename with ambiguity detection
5. Keep full rebuild path as maintenance fallback.
6. Ensure persistence and recovery behavior:
   - save FAISS index atomically via temp-file replace
   - rebuild FAISS from catalog text if FAISS/catalog counts diverge
7. Add read-optimized search runtime mode:
   - keep mutable CPU source-of-truth index for add/update/delete correctness
   - support optional GPU runtime search index via `RAG_FAISS_SEARCH_DEVICE=gpu`
   - fall back to CPU search runtime index automatically when GPU path is unavailable
8. Update API surface:
   - keep `DELETE /api/documents/{filename}`
   - add preferred `DELETE /api/document?source_path=...`
9. Update dashboard implementation:
   - move inline dashboard HTML/JS out of `app.py` into `templates/dashboard.html`
   - ensure dashboard delete actions call `DELETE /api/document?source_path=...`
10. Update docs and return payloads:
   - return `operation`, `added_chunks`, `removed_chunks`, `skipped`

## Constraints
- Do not regress existing search endpoint behavior.
- Do not force ingestion to only `./documents`.
- Keep changes minimal and focused.
- Preserve CPU fallback behavior if GPU/FAISS GPU capabilities are uncertain.

## Validation checklist
- Add same file twice unchanged -> second call returns skipped/noop.
- Add file changed in-place -> returns update and only this file is re-embedded.
- Delete by source path removes only that file's vectors.
- Duplicate filename across different folders -> filename delete is ambiguous (409 path in API), source path delete succeeds.
- Restart process and list/search still work from persisted FAISS+catalog.
- `/dashboard` serves HTML template from `templates/dashboard.html`.
- Dashboard HTML references `/api/document?source_path=` for delete operations.
- Docker test run passes: `docker compose run --rm --entrypoint python faiss-rag-server -m unittest discover -s tests -v`.

## Deliverables
- Code updates in `rag/index.py`, `rag/docHandlers.py`, `api/api_endpoints.py`
- New `rag/catalog.py`
- Dashboard template file: `templates/dashboard.html`
- Dashboard route updates in `app.py`
- Test files: `tests/test_incremental_indexing.py`, `tests/test_dashboard.py`
- Runtime config update in `docker-compose.yml` for `RAG_FAISS_SEARCH_DEVICE`
- Updated `README.md`
- Updated implementation checklist in `INCREMENTAL_INDEXING_PLAN.md`
- Updated progress log in `WORK_PROGRESS_INCREMENTAL_INDEXING.md`

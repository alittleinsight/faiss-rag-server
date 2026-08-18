# Incremental Embedding + Indexing Plan (File-by-File)

## Objective
Move from full in-memory rebuilds to incremental add/update/delete by file so:
- adding one file only embeds and indexes that file's chunks
- deleting one file only removes that file's chunk vectors
- full rebuild becomes a maintenance operation, not the default path

Path scope:
- indexed files are not restricted to `./documents`; any file path that is reachable by the runtime is supported
- paths are canonicalized to absolute resolved `source_path` before dedupe, update, or delete logic

## Current Gap (in this repo)
Today, `rag_add_document` and `rag_delete_document` both eventually call `rebuild_index_from_memory(...)`, which:
1. re-embeds all texts
2. recreates the FAISS index from scratch

This is the behavior we are replacing.

## Target Design

### 1) Stable Vector IDs (required for targeted delete)
Assign every chunk a stable int64 `chunk_id`.
- Use monotonic ID generation persisted on disk (no random IDs).
- Store mapping metadata for each `chunk_id`.

Suggested metadata fields per chunk:
- `chunk_id`
- `source_path` (absolute canonical path)
- `source_hash` (file content hash)
- `chunk_hash` (hash of chunk content)
- `chunk_order` (0..N-1 per file)
- loader metadata (page, etc.)
- `created_at`, `updated_at`

### 2) Persisted Catalog (SQLite)
Add `vectorstore/catalog.db` for index bookkeeping.

Suggested tables:
- `documents(source_path PRIMARY KEY, source_hash, mtime, size, chunk_count, updated_at)`
- `chunks(chunk_id PRIMARY KEY, source_path, chunk_order, chunk_hash, metadata_json, FOREIGN KEY source_path)`
- `state(key PRIMARY KEY, value)` for counters like `next_chunk_id`

Benefits:
- fast lookup of all chunk IDs for a file
- idempotent add/update behavior
- durable mapping independent of process memory

### 3) FAISS Index Type and ID Management
Wrap the index with `faiss.IndexIDMap2` so vectors are added with explicit IDs and removed by ID.

- Build base index as today (CPU/GPU aware), then wrap in ID map:
  - `id_index = faiss.IndexIDMap2(base_index)`
- Add vectors via `id_index.add_with_ids(vectors, ids)`
- Delete vectors via `id_index.remove_ids(faiss.IDSelectorBatch(ids))`

Notes:
- Keep `DEFAULT_FAISS_DIM` and embedding model unchanged initially.
- For GPU mode, keep compute on GPU but persist CPU copy.
- If the specific IVF/GPU combo shows remove limitations, do staged fallback:
  1. run mutable CPU index for correctness
  2. optional periodic rebuild to optimized GPU IVF snapshot for search

### 4) Per-File Operations

#### Add file
1. Load and split only that file.
2. Compute file hash.
3. If same hash exists in `documents`, skip (already indexed).
4. Embed only new chunks.
5. Allocate chunk IDs, add to FAISS with IDs.
6. Upsert document and chunk records.
7. Persist FAISS + catalog atomically.

#### Delete file
1. Resolve canonical `source_path`.
2. Query all chunk IDs for that file from catalog.
3. Remove IDs from FAISS.
4. Delete rows from `chunks` and `documents`.
5. Persist FAISS + catalog atomically.

#### Update file (same path, changed content)
1. Compare hash/mtime.
2. If changed: remove existing chunk IDs for file.
3. Re-split + re-embed only that file.
4. Add new vectors/metadata.

### 5) Startup / Recovery
On startup:
1. Load FAISS index and SQLite catalog.
2. Validate consistency:
   - catalog chunk count vs FAISS ntotal
   - optional random spot checks of IDs
3. If inconsistent, trigger controlled recovery:
   - attempt repair from catalog + files
   - fallback to full rebuild only when necessary

### 6) API Surface Changes
Preserve existing endpoints and behavior where possible:
- `POST /api/add` -> incremental add/update for one source path
- `DELETE /api/documents/{filename}` -> keep for backward compatibility, but treat as ambiguous and return 409 when filename matches multiple source paths
- preferred delete operation should use source path identity, e.g. `DELETE /api/document?source_path=...` (or `POST /api/delete` with `source_path`)
- optional new endpoint: `POST /api/sync` to detect changed/removed files for a configured root set (not only `documents/`)

Response additions (recommended):
- `added_chunks`
- `removed_chunks`
- `skipped` + reason
- `operation`: `add|update|delete|noop`

### 7) Concurrency + Atomicity
- Add a process-level write lock around mutating operations (add/delete/update).
- Use temp files + replace for FAISS persistence:
  - write `index.faiss.tmp` then atomic rename.
- Wrap SQLite mutations in transactions.
- Commit catalog and FAISS save as one operation boundary (best effort with rollback strategy).

### 8) Migration Plan (phased)

#### Phase 0: Guardrails + tests (short)
- Add tests that assert current behavior for add/delete/search endpoints.
- Add benchmark baseline for add/delete latency and index size.

#### Phase 1: Catalog + ID plumbing
- Introduce catalog module and chunk ID generation.
- Extend in-memory metadata to include `chunk_id`.
- Keep existing rebuild flow temporarily.

#### Phase 2: Mutable FAISS path
- Introduce `IndexIDMap2` and `add_with_ids` / `remove_ids` APIs.
- Implement incremental add/delete without global re-embed.
- Keep full rebuild command as fallback.

#### Phase 3: Update + sync
- Add file hash detection and update operation.
- Implement optional directory sync endpoint/job.

#### Phase 4: Optimization + compaction
- Add periodic compaction/repack command if delete churn is high.
- Evaluate IVF params and retraining thresholds after many mutations.

### 9) Testing Strategy
- Unit tests:
  - add one file adds only its chunks
  - delete one file removes only its chunks
  - update one file replaces only its chunk IDs
- Integration tests:
  - restart process and verify persistence
  - mixed add/delete sequences maintain correct retrieval
- Regression tests:
  - search quality unchanged for unchanged corpus

### 10) Rollback Strategy
- Keep `rebuild_index_from_disk()` as emergency command.
- Feature flag incremental mode:
  - `RAG_INCREMENTAL_INDEXING=true|false`
- If incremental path fails, log and fallback to rebuild with clear warning.

## Implementation Checklist
- [x] Add `rag/catalog.py` (SQLite schema + CRUD)
- [x] Add ID-aware index wrapper in `rag/index.py`
- [x] Replace `rebuild_index_from_memory` usage in add/delete flows
- [x] Add hash-based update detection
- [x] Add write lock and atomic persistence helpers
- [x] Add tests for add/delete/update without full rebuild
- [x] Add migration notes to README

## Recent Changes
- Extracted dashboard markup/scripts from `app.py` into `templates/dashboard.html` for readability and easier maintenance.
- Updated dashboard delete action to use exact source-path deletion: `DELETE /api/document?source_path=...`.
- Added dashboard coverage in `tests/test_dashboard.py`:
  - verifies `/dashboard` serves HTML
  - verifies dashboard references `/api/document?source_path=`
- Validated test suite in Docker:
  - `docker compose run --rm --entrypoint python faiss-rag-server -m unittest discover -s tests -v`
  - latest result: `Ran 5 tests ... OK`
- Added read-optimized hybrid search runtime mode in `rag/index.py`:
  - mutable source-of-truth index remains CPU (`IndexIDMap2(IndexFlatIP)`) for safe add/update/delete
  - optional GPU search runtime index enabled via `RAG_FAISS_SEARCH_DEVICE=gpu`
  - automatic fallback to CPU search runtime index when GPU/FAISS GPU APIs are unavailable
- Added compose/runtime configuration and docs for `RAG_FAISS_SEARCH_DEVICE`.

## Success Criteria
- Add/delete of a single file completes without re-embedding unaffected files.
- FAISS `ntotal` changes only by that file's chunk delta.
- Restart preserves exact mapping between vectors and source metadata.
- Full rebuild is no longer required for routine file lifecycle operations.

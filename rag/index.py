# rag/index.py

import hashlib
import json
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from .catalog import (
    canonicalize_source_path,
    delete_document_and_chunks,
    get_all_chunks,
    get_chunk_ids_for_source_path,
    get_document,
    get_source_paths_for_filename,
    init_catalog,
    insert_chunk_rows,
    list_documents,
    next_chunk_ids,
    upsert_document,
)
from .embeddings import cleanup_after_embedding, embeddings
from . import embeddings as embeddings_module

DOCUMENTS_PATH = Path("./documents")
VECTORSTORE_PATH = Path("./vectorstore")
VECTORSTORE_PATH.mkdir(exist_ok=True)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3
DEFAULT_FAISS_DIM = int(os.getenv("DEFAULT_FAISS_DIM", "1024"))
RAG_FAISS_SEARCH_DEVICE = os.getenv("RAG_FAISS_SEARCH_DEVICE", "cpu").strip().lower()
EMBEDDING_GPU_INDEX = int(getattr(embeddings_module, "gpuIndex", 0))
EMBEDDING_USE_GPU = bool(getattr(embeddings_module, "useGpu", False))

SUPPORTED_TEXT_SUFFIXES = {".txt", ".py", ".js", ".ts", ".json", ".csv", ".html", ".log"}
CATALOG_PATH = VECTORSTORE_PATH / "catalog.db"
FAISS_PATH = VECTORSTORE_PATH / "index.faiss"
LEGACY_PKL_PATH = VECTORSTORE_PATH / "index.pkl"

index: Optional[Any] = None
search_runtime_index: Optional[Any] = None
chunk_store: dict[int, dict] = {}
_gpu_resources = None

_WRITE_LOCK = threading.Lock()


def _new_mutable_index() -> Any:
    # IndexIDMap2 enables add/remove by explicit int64 IDs.
    return faiss.IndexIDMap2(faiss.IndexFlatIP(DEFAULT_FAISS_DIM))


def _supports_faiss_gpu() -> bool:
    required_symbols = ["StandardGpuResources", "index_cpu_to_gpu"]
    return all(hasattr(faiss, symbol) for symbol in required_symbols)


def _refresh_search_runtime_index() -> None:
    global search_runtime_index, _gpu_resources

    search_runtime_index = index
    if index is None:
        return

    if RAG_FAISS_SEARCH_DEVICE != "gpu":
        return

    if not EMBEDDING_USE_GPU:
        print("RAG_FAISS_SEARCH_DEVICE=gpu requested, but embeddings are not using GPU; keeping CPU search index")
        return

    if not _supports_faiss_gpu():
        print("RAG_FAISS_SEARCH_DEVICE=gpu requested, but FAISS GPU APIs are unavailable; keeping CPU search index")
        return

    try:
        _gpu_resources = faiss.StandardGpuResources()
        search_runtime_index = faiss.index_cpu_to_gpu(_gpu_resources, EMBEDDING_GPU_INDEX, index)
        print(f"Search runtime index is on FAISS GPU {EMBEDDING_GPU_INDEX}")
    except Exception as exc:
        search_runtime_index = index
        print(f"Failed to move search runtime index to GPU: {exc}; using CPU search index")


def _save_index() -> None:
    if index is None:
        return

    tmp_path = FAISS_PATH.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(tmp_path))
    tmp_path.replace(FAISS_PATH)


def _is_id_map_index(candidate_index: Any) -> bool:
    return "IndexIDMap" in str(type(candidate_index))


def _normalize_embedding_matrix(vectors) -> np.ndarray:
    arr = np.array(vectors, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _load_docs_for_source_path(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(file_path)).load()
    elif suffix == ".md":
        docs = UnstructuredMarkdownLoader(str(file_path)).load()
    elif suffix in SUPPORTED_TEXT_SUFFIXES:
        docs = TextLoader(str(file_path), encoding="utf-8").load()
    else:
        return None, f"Unsupported file type: '{suffix}'"

    source_path = canonicalize_source_path(str(file_path))
    for doc in docs:
        doc.metadata = dict(doc.metadata or {})
        doc.metadata["source"] = source_path

    return docs, None


def _split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def _file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _remove_ids_from_index(ids: list[int]) -> int:
    if index is None or not ids:
        return 0
    id_array = np.array(ids, dtype="int64")
    removed = int(index.remove_ids(id_array))
    return removed


def _clear_and_rebuild_faiss_from_catalog() -> None:
    global index, chunk_store
    index = _new_mutable_index()
    chunk_store = {}

    chunk_rows = get_all_chunks(CATALOG_PATH)
    if not chunk_rows:
        _save_index()
        return

    texts = [row["text"] for row in chunk_rows]
    ids = np.array([int(row["chunk_id"]) for row in chunk_rows], dtype="int64")
    vectors = _normalize_embedding_matrix(embeddings.embed_documents(texts))
    cleanup_after_embedding()

    index.add_with_ids(vectors, ids)
    for row in chunk_rows:
        chunk_store[int(row["chunk_id"])] = {
            "text": row["text"],
            "metadata": row["metadata"],
        }

    _save_index()
    _refresh_search_runtime_index()


def _import_legacy_pickle_into_catalog() -> int:
    if not LEGACY_PKL_PATH.exists():
        return 0

    try:
        with LEGACY_PKL_PATH.open("rb") as handle:
            data = pickle.load(handle)
    except Exception as exc:
        print(f"Legacy metadata migration skipped: failed to read index.pkl: {exc}")
        return 0

    texts = data.get("texts", []) if isinstance(data, dict) else []
    metadatas = data.get("metadatas", []) if isinstance(data, dict) else []
    if not texts or len(texts) != len(metadatas):
        return 0

    grouped = {}
    for text, metadata in zip(texts, metadatas):
        metadata = dict(metadata or {})
        source_path = metadata.get("source")
        if not source_path:
            source_path = "legacy://unknown-source"
        canonical_source = canonicalize_source_path(source_path) if "://" not in source_path else source_path
        grouped.setdefault(canonical_source, []).append((text, metadata))

    imported_chunks = 0
    for source_path, items in grouped.items():
        ids = next_chunk_ids(CATALOG_PATH, len(items))
        rows = []
        for chunk_order, (text, metadata) in enumerate(items):
            chunk_id = int(ids[chunk_order])
            metadata = dict(metadata or {})
            metadata["source"] = source_path
            metadata["chunk_id"] = chunk_id
            metadata["chunk_order"] = chunk_order
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "source_path": source_path,
                    "chunk_order": chunk_order,
                    "chunk_hash": _chunk_hash(text),
                    "text": text,
                    "metadata": metadata,
                }
            )

        upsert_document(
            CATALOG_PATH,
            source_path=source_path,
            source_hash="legacy-import",
            mtime=0.0,
            size=0,
            chunk_count=len(rows),
        )
        insert_chunk_rows(CATALOG_PATH, rows)
        imported_chunks += len(rows)

    if imported_chunks > 0:
        print(f"Migrated {imported_chunks} legacy chunks from index.pkl into catalog")

    return imported_chunks


def load_or_rebuild_index() -> None:
    global index, chunk_store
    init_catalog(CATALOG_PATH)

    if FAISS_PATH.exists():
        print("Loading existing FAISS index from disk...")
        index = faiss.read_index(str(FAISS_PATH))
    else:
        index = _new_mutable_index()

    if index is not None and not _is_id_map_index(index):
        print("Loaded FAISS index is not ID-mutable. Replacing with IndexIDMap2 and rebuilding from catalog.")
        index = _new_mutable_index()

    chunk_store = {
        int(row["chunk_id"]): {
            "text": row["text"],
            "metadata": row["metadata"],
        }
        for row in get_all_chunks(CATALOG_PATH)
    }

    if not chunk_store:
        _import_legacy_pickle_into_catalog()
        chunk_store = {
            int(row["chunk_id"]): {
                "text": row["text"],
                "metadata": row["metadata"],
            }
            for row in get_all_chunks(CATALOG_PATH)
        }

    if index is None:
        index = _new_mutable_index()

    if int(index.ntotal) != len(chunk_store):
        print("FAISS/catalog mismatch detected. Rebuilding FAISS from catalog text...")
        _clear_and_rebuild_faiss_from_catalog()

    _refresh_search_runtime_index()


def upsert_file_to_index(source_path: str) -> dict:
    global index, chunk_store
    with _WRITE_LOCK:
        candidate_path = Path(source_path).expanduser()
        if not candidate_path.exists():
            return {
                "status": "error",
                "message": f"Document not found at path: {candidate_path}",
            }
        if not candidate_path.is_file():
            return {
                "status": "error",
                "message": f"Path is not a file: {candidate_path}",
            }

        canonical_source_path = canonicalize_source_path(str(candidate_path))
        source_hash = _file_hash(candidate_path)
        existing_doc = get_document(CATALOG_PATH, canonical_source_path)

        if existing_doc and existing_doc.get("source_hash") == source_hash:
            docs = list_documents(CATALOG_PATH)
            return {
                "status": "success",
                "skipped": True,
                "operation": "noop",
                "message": "Document already indexed for this source_path",
                "added_source_path": canonical_source_path,
                "indexed_documents": len(docs),
                "indexed_chunks": len(chunk_store),
            }

        loaded_docs, load_error = _load_docs_for_source_path(candidate_path)
        if load_error:
            return {
                "status": "error",
                "message": load_error,
            }

        chunks = _split_documents(loaded_docs)
        if not chunks:
            return {
                "status": "error",
                "message": f"No indexable content found in file: {candidate_path}",
            }

        removed_chunks = 0
        if existing_doc:
            stale_ids = get_chunk_ids_for_source_path(CATALOG_PATH, canonical_source_path)
            removed_chunks = _remove_ids_from_index(stale_ids)
            for stale_id in stale_ids:
                chunk_store.pop(int(stale_id), None)
            delete_document_and_chunks(CATALOG_PATH, canonical_source_path)

        texts = [chunk.page_content for chunk in chunks]
        vectors = _normalize_embedding_matrix(embeddings.embed_documents(texts))
        cleanup_after_embedding()

        ids = next_chunk_ids(CATALOG_PATH, len(chunks))
        ids_array = np.array(ids, dtype="int64")
        index.add_with_ids(vectors, ids_array)

        chunk_rows = []
        for chunk_id, chunk_order, chunk in zip(ids, range(len(chunks)), chunks):
            metadata = dict(chunk.metadata or {})
            metadata["source"] = canonical_source_path
            metadata["chunk_id"] = int(chunk_id)
            metadata["chunk_order"] = int(chunk_order)

            text = chunk.page_content
            chunk_rows.append(
                {
                    "chunk_id": int(chunk_id),
                    "source_path": canonical_source_path,
                    "chunk_order": int(chunk_order),
                    "chunk_hash": _chunk_hash(text),
                    "text": text,
                    "metadata": metadata,
                }
            )
            chunk_store[int(chunk_id)] = {
                "text": text,
                "metadata": metadata,
            }

        stat = candidate_path.stat()
        upsert_document(
            CATALOG_PATH,
            source_path=canonical_source_path,
            source_hash=source_hash,
            mtime=stat.st_mtime,
            size=stat.st_size,
            chunk_count=len(chunks),
        )
        insert_chunk_rows(CATALOG_PATH, chunk_rows)
        _save_index()
        _refresh_search_runtime_index()

        docs = list_documents(CATALOG_PATH)
        return {
            "status": "success",
            "operation": "update" if existing_doc else "add",
            "added_source_path": canonical_source_path,
            "added_chunks": len(chunks),
            "removed_chunks": int(removed_chunks),
            "indexed_documents": len(docs),
            "indexed_chunks": len(chunk_store),
        }


def delete_document_by_source_path(source_path: str) -> dict:
    global chunk_store
    with _WRITE_LOCK:
        canonical_source_path = canonicalize_source_path(source_path)
        ids = get_chunk_ids_for_source_path(CATALOG_PATH, canonical_source_path)
        if not ids:
            return {
                "status": "not_found",
                "source_path": canonical_source_path,
                "chunks_removed": 0,
            }

        removed = _remove_ids_from_index(ids)
        for chunk_id in ids:
            chunk_store.pop(int(chunk_id), None)
        delete_document_and_chunks(CATALOG_PATH, canonical_source_path)
        _save_index()
        _refresh_search_runtime_index()

        return {
            "status": "success",
            "operation": "delete",
            "source_path": canonical_source_path,
            "chunks_removed": int(removed),
            "index_only_delete": True,
        }


def delete_document_by_filename(filename: str) -> dict:
    matching_paths = get_source_paths_for_filename(CATALOG_PATH, filename)
    if not matching_paths:
        return {
            "status": "not_found",
            "filename": filename,
            "chunks_removed": 0,
        }

    if len(matching_paths) > 1:
        return {
            "status": "ambiguous",
            "filename": filename,
            "matches": matching_paths,
            "message": "Filename matches multiple source paths. Delete by source_path instead.",
        }

    result = delete_document_by_source_path(matching_paths[0])
    result["filename"] = filename
    return result


def get_chunk_count() -> int:
    return len(chunk_store)


def search_index(query_vector: np.ndarray, k: int):
    active_index = search_runtime_index or index
    if active_index is None or get_chunk_count() == 0:
        return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")
    return active_index.search(query_vector, k)


def get_chunk_record(chunk_id: int) -> Optional[dict]:
    return chunk_store.get(int(chunk_id))


def get_documents_summary() -> dict:
    docs = list_documents(CATALOG_PATH)
    return {
        "documents": docs,
        "count": len(docs),
    }


def rebuild_index_from_disk() -> dict:
    docs_rebuilt = 0
    chunks_rebuilt = 0
    if DOCUMENTS_PATH.exists():
        for file_path in DOCUMENTS_PATH.rglob("*"):
            if not file_path.is_file():
                continue
            result = upsert_file_to_index(str(file_path))
            if result.get("status") == "success":
                docs_rebuilt += 1
                chunks_rebuilt += int(result.get("added_chunks", 0))

    return {
        "status": "success",
        "operation": "bulk_upsert",
        "documents_processed": docs_rebuilt,
        "chunks_added": chunks_rebuilt,
        "indexed_chunks": get_chunk_count(),
    }


def export_catalog_snapshot() -> str:
    snapshot = {
        "chunk_count": get_chunk_count(),
        "documents": list_documents(CATALOG_PATH),
    }
    return json.dumps(snapshot, indent=2)


load_or_rebuild_index()

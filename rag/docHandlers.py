import numpy as np

from pathlib import Path
from .index import TOP_K
from . import index as index_module

_documents_cache = {
    "key": None,
    "value": None,
}


# Helper: centralize retrieval logic for RAG
def rag_retrieve_context(query: str, k: int = TOP_K):
    if not index_module.index or index_module.get_chunk_count() == 0:
        return [], [], []

    q_vec = np.array(index_module.embeddings.embed_query(query)).astype("float32").reshape(1, -1)

    k = min(k, index_module.get_chunk_count())
    D, I = index_module.search_index(q_vec, k)

    results = []
    indices = []
    scores = []

    for score, chunk_id in zip(D[0], I[0]):
        chunk_id = int(chunk_id)
        record = index_module.get_chunk_record(chunk_id)
        if chunk_id >= 0 and record is not None:
            indices.append(chunk_id)
            scores.append(float(score))
            results.append({
                "text": record["text"],
                "index": chunk_id,
                "score": float(score),
                "metadata": record["metadata"],
            })

    return results, indices, scores

# return list of indexed document filenames
def rag_list_documents():
    docs = index_module.get_documents_summary()
    cache_key = tuple((item["source_path"], item.get("chunk_count", 0)) for item in docs["documents"])
    if _documents_cache["key"] == cache_key and _documents_cache["value"] is not None:
        return _documents_cache["value"]

    result = docs
    _documents_cache["key"] = cache_key
    _documents_cache["value"] = result

    return result

def rag_add_document(source_path: str | None = None):
    if source_path:
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
        return index_module.upsert_file_to_index(str(candidate_path))
    else:
        return index_module.rebuild_index_from_disk()



def rag_delete_document(filename: str):
    return index_module.delete_document_by_filename(filename)


def rag_delete_document_by_source_path(source_path: str):
    return index_module.delete_document_by_source_path(source_path)
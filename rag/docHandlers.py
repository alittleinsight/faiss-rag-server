import numpy as np

from pathlib import Path
from .embeddings import embeddings
from .index import rebuild_index_from_disk, rebuild_index_from_memory, DOCUMENTS_PATH, TOP_K
from . import index as index_module


# Helper: centralize retrieval logic for RAG
def retrieve_context(query: str, k: int = TOP_K):
    if not index_module.index or len(index_module.texts) == 0:
        return [], [], []

    q_vec = np.array(embeddings.embed_query(query)).astype("float32").reshape(1, -1)

    k = min(k, len(index_module.texts))
    D, I = index_module.index.search(q_vec, k)

    results = []
    indices = []
    scores = []

    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(index_module.texts):
            indices.append(int(idx))
            scores.append(float(score))
            results.append({
                "text": index_module.texts[idx],
                "index": int(idx),
                "score": float(score),
                "metadata": index_module.metadatas[idx]
            })

    return results, indices, scores

# return list of indexed document filenames
def rag_list_documents():
    return {"documents": sorted({Path(m.get("source", "")).name for m in index_module.metadatas if "source" in m})}

def rag_upload_document():
    rebuild_index_from_disk()
    # ToDo update return with more useful info
    return {"status": "success"}

def rag_delete_document(filename: str):
    #global index, texts, metadatas
    
    # 1. Find and remove the physical file
    file_path = None
    for p in DOCUMENTS_PATH.rglob(filename):
        if p.is_file():
            file_path = p
            break
    
    if file_path:
        try:
            file_path.unlink()  # ← PHYSICALLY DELETE THE FILE
            print(f"Deleted physical file: {file_path}")
        except Exception as e:
            print(f"Warning: Could not delete file {file_path}: {e}")
    else:
        print(f"File {filename} not found in documents folder (maybe already deleted)")

    # 2. Remove its chunks from the index
    new_texts, new_metadatas = [], []
    removed = 0
    for t, m in zip(index_module.texts, index_module.metadatas):
        if Path(m.get("source", "")).name == filename:
            removed += 1
            continue  # skip this chunk
        new_texts.append(t)
        new_metadatas.append(m)

    if removed > 0:
        # 3. Rebuild index from remaining chunks
        #texts, metadatas = new_texts, new_metadatas
        rebuild_index_from_memory(new_texts, new_metadatas)

    return {"deleted": filename, "chunks_removed": removed, "file_removed": bool(file_path)}
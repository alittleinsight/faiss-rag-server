from pathlib import Path
from fastapi import APIRouter, Form, HTTPException, Query

from rag.docHandlers import (
    rag_add_document,
    rag_delete_document,
    rag_delete_document_by_source_path,
    rag_list_documents,
    rag_retrieve_context,
)
from rag.index import TOP_K

router = APIRouter()


def _resolve_source_path_for_container(source_path: str) -> Path:
    normalized = source_path.strip().replace("\\", "/")
    workspace_root = Path("/workspace")

    candidates: list[Path] = []
    raw_candidate = Path(normalized).expanduser()
    candidates.append(raw_candidate)

    if not raw_candidate.is_absolute():
        candidates.append((workspace_root / normalized).expanduser())

    repo_marker = "/faiss-rag-server/"
    lower = normalized.lower()
    marker_pos = lower.find(repo_marker)
    if marker_pos != -1:
        repo_relative = normalized[marker_pos + len(repo_marker):]
        candidates.append(workspace_root / repo_relative)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return raw_candidate

# -----------------------------API endpoints ---------------------------------------------
@router.post("/api/add")
async def add_document(
    source_path: str | None = Form(default=None),
):
    if not source_path:
        raise HTTPException(400, "source_path is required; file payloads are disabled to avoid copying files")

    if source_path:
        candidate_path = _resolve_source_path_for_container(source_path)
        if not candidate_path.exists():
            raise HTTPException(
                404,
                (
                    f"Document not found at path: {source_path}. "
                    "For Docker use container-visible paths like '/workspace/documents/your-file.pdf' "
                    "or relative 'documents/your-file.pdf'."
                ),
            )
        if not candidate_path.is_file():
            raise HTTPException(400, f"Path is not a file: {candidate_path}")

        result = rag_add_document(source_path=str(candidate_path))
        if result.get("status") != "success":
            raise HTTPException(400, result.get("message", "Failed to index file from source path"))
        return result

@router.get("/api/documents")
def list_documents():
    return rag_list_documents()

@router.delete("/api/documents/{filename}")
def delete_document(filename: str):

    result = rag_delete_document(filename)
    if result.get("status") == "ambiguous":
        raise HTTPException(409, result.get("message", "Filename is ambiguous; delete by source_path"))
    if result.get("chunks_removed", 0) == 0:
        raise HTTPException(404, "File not found in index")
    return result


@router.delete("/api/document")
def delete_document_by_source_path(source_path: str = Query(...)):
    result = rag_delete_document_by_source_path(source_path)
    if result.get("chunks_removed", 0) == 0:
        raise HTTPException(404, "Document source_path not found in index")
    return result

@router.get("/api/search")
def search(q: str, k: int = TOP_K):

    results, indices, scores = rag_retrieve_context(q, k)

    if not results:
        return {"results": []}

    # Build improved structured JSON results
    structured_results = []
    for i, r in enumerate(results):
        meta = r.get("metadata", {})
        structured_results.append({
            "rank": i + 1,
            "score": scores[i],
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "unknown"),
            "text": r["text"],
            "metadata": meta
        })
    
    return {
        "query": q,
        "top_k": k,
        "results": structured_results,     # list of {text, index, score, metadata}
        #"indices": indices,     # list[int]
        #"scores": scores        # list[float]
    }
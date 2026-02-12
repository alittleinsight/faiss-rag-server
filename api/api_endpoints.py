import anyio

from pathlib import Path
from unittest import result
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File, HTTPException

from rag.docHandlers import rag_retrieve_context, rag_list_documents, rag_upload_document, rag_delete_document, DOCUMENTS_PATH
from rag.index import TOP_K

router = APIRouter()

# -----------------------------API endpoints ---------------------------------------------
@router.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    save_path = DOCUMENTS_PATH / file.filename
    if save_path.exists():
        stem, suffix = save_path.stem, save_path.suffix
        i = 1
        while (save_path := DOCUMENTS_PATH / f"{stem}_{i}{suffix}").exists():
            i += 1
    content = await file.read()
    async with await anyio.open_file(save_path, "wb") as f:
        await f.write(content)
    rag_upload_document()

    return {"status": "success", "filename": save_path.name}

@router.get("/api/documents")
def list_documents():
    return rag_list_documents()

@router.delete("/api/documents/{filename}")
def delete_document(filename: str):

    result = rag_delete_document(filename)
    if result.get("chunks_removed", 0) == 0:
        raise HTTPException(404, "File not found in index")
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
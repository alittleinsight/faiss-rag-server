import anyio

from pathlib import Path
from unittest import result
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File, HTTPException

from rag.docHandlers import retrieve_context, rag_list_documents, rag_upload_document, rag_delete_document, DOCUMENTS_PATH

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
def search(q: str, k: int):

    results, indices, scores = retrieve_context(q, k)

    if not results:
        return {"results": []}

    return {
        "query": q,
        "top_k": k,
        "results": results,     # list of {text, index, score, metadata}
        "indices": indices,     # list[int]
        "scores": scores        # list[float]
    }
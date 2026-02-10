import asyncio
from typing import Any, Dict
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .tool_definitions import TOOL_DEFINITIONS
from .tool_registry import TOOL_REGISTRY, register_tool

from rag.docHandlers import retrieve_context
#from rag.models import SERVER_NAME, SERVER_VERSION

SERVER_NAME = "faiss-rag-server"
SERVER_VERSION = "1.0.0"

router = APIRouter()

# ---------- SSE ROOT (GET /) ----------

@router.get("/mcp")
async def vscode_sse_root():
    async def event_stream():
        # Initial connection event
        yield 'data: {"type":"connected"}\n\n'
        while True:
            # Heartbeat
            yield 'data: {"type":"ping"}\n\n'
            await asyncio.sleep(15)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Access-Control-Allow-Origin": "*"}
    )


# ---------- MCP DISPATCHER (POST /) ----------

@router.post("/mcp")
async def vscode_mcp_root(req: Dict[str, Any]):
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")

    # initialize
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                },
                "capabilities": {
                    "tools": {
                        "list": {},
                        "call": {}
                },
                #"resources": None,
                "roots": {
                    "listChanged": True
                },
                "sampling": {},
                "elicitation": {
                    "form": {},
                    "url": {}
                },
                "tasks": {
                    "list": {},
                    "cancel": {},
                    "run": {},
                    "progress": {},
                    "requests": {
                        "sampling": {
                            "createMessage": {}
                        },
                        "elicitation": {
                            "create": {}
                        }
                    }
                }
            }
        }
    }

    # tools/list
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOL_DEFINITIONS["tools"]
            }
        }

    # tools/call
    if method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        handler = TOOL_REGISTRY.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
                }
            }

        output = handler(tool_args)

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": output
        }

    if method == "notifications/initialized":
        return JSONResponse(status_code=200, content=None)

    if method == "notifications/cancelled": # Acknowledge cancellation but do nothing 
        return { 
            "jsonrpc": "2.0", 
            "id": None, 
            "result": None 
        }    

    # unknown
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32601,
            "message": f"Unknown method: {method}"
        }
    }

# ---------- TOOL IMPLEMENTATIONS (VS Code path) ----------
@register_tool("retrieve_context")
def tool_retrieve_context(params: Dict[str, Any]) -> Dict[str, Any]:
    query = params.get("query")
    top_k = params.get("top_k", 5)

    # Validate input
    if not isinstance(query, str):
        return {
            "content": [
                {"type": "text", "text": "Error: missing or invalid 'query' field"}
            ]
        }

    if not isinstance(top_k, int):
        try:
            top_k = int(top_k)
        except Exception:
            return {
                "content": [
                    {"type": "text", "text": "Error: 'top_k' must be an integer"}
                ]
            }

    # Retrieve FAISS results
    results, indices, scores = retrieve_context(query, top_k)

    # Build summary text
    summary_lines = []
    for i, r in enumerate(results):
        snippet = r["text"].strip().replace("\n", " ")
        snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
        summary_lines.append(f"Chunk {i+1}: {snippet}")

    summary_text = (
        f"Retrieved {len(results)} relevant chunks for query: '{query}'.\n\n"
        + "\n".join(summary_lines)
    )

    # Build sources text
    sources = []
    for r in results:
        meta = r.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", "unknown")
        sources.append(f"- {source}, page {page}")

    full_text = "\n\n".join(
        f"Chunk {i+1}:\n{r['text']}"
        for i, r in enumerate(results)
    )

    sources_text = "\n".join(
        f"- {r.get('metadata', {}).get('source', 'unknown')} "
        f"(page {r.get('metadata', {}).get('page', 'unknown')})"
        for r in results
    )

    json_block = json.dumps({
        "query": query,
        "top_k": top_k,
        "results": results,
        "indices": indices,
        "scores": scores
    }, indent=2)

    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"Retrieved {len(results)} chunks for query: '{query}'.\n\n"
                    f"{full_text}\n\n"
                    f"Sources:\n{sources_text}"
                )
            },
            {
                "type": "text",
                "text": json_block
            }
        ]
    }


#@register_tool("retrieve_context")
def old_tool_retrieve_context(params: Dict[str, Any]) -> Dict[str, Any]:
    query = params.get("query")
    top_k = params.get("top_k", 5)

    # Validate types for static analysis + runtime safety
    if not isinstance(query, str):
        return {
            "type": "error",
            "error": "Invalid or missing 'query' field"
        }
    if not isinstance(top_k, int):
        try:
            top_k = int(top_k)
        except Exception:
            return {
                "type": "error",
                "error": "'top_k' must be an integer"
            }
        
    # Now Pylance knows query is str and top_k is int
    results, indices, scores = retrieve_context(query, top_k)

    return {
        "query": query,
        "top_k": top_k,
        "results": results,     # list of {text, index, score, metadata}
        "indices": indices,     # list[int]
        "scores": scores        # list[float]
    }


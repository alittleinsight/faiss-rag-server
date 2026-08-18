from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path

from mcp import mcp_router as mcp_router
from api import api_router as api_router

# ------------------- Globals -------------------
app = FastAPI(title="FAISS RAG Server", openapi_url="/openapi.json")

# MCP-over-HTTP
app.include_router(mcp_router)
app.include_router(api_router)

# In-memory queue for MCP plugin (LM Studio)
app.state.last_user_query = None


@app.get("/health")
async def health():
    return {"status": "ok"}

# ------------------- Dashboard -------------------
@app.get("/dashboard", response_class=FileResponse)
async def dashboard():
    dashboard_path = Path(__file__).resolve().parent / "templates" / "dashboard.html"
    return FileResponse(str(dashboard_path))
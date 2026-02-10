import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

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
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return """
    <!DOCTYPE html>
    <html><head><title>Local RAG Server</title><meta charset="utf-8"><style>
      body{font-family:system-ui;max-width:900px;margin:40px auto;padding:20px;background:#fafafa}
      h1{color:#2c7}a{color:#07a;text-decoration:none}
      .box{background:#fff;padding:24px;border-radius:12px;margin:24px 0;box-shadow:0 4px 12px rgba(0,0,0,0.05)}
      code{background:#f0f0f0;padding:12px 16px;display:block;font-size:1.1em;border-radius:8px}
      table{width:100%;border-collapse:collapse;margin-top:16px}
      th, td{padding:12px;text-align:left;border-bottom:1px solid #eee}
      th{background:#f7f7f7}
      .btn{padding:10px 20px;border:none;border-radius:8px;font-size:15px;cursor:pointer;margin:8px 8px 8px 0;font-weight:600}
      .btn-upload{background:#2c7;color:white}
      .btn-upload:hover{background:#1a5}
      .btn-delete{background:#c33;color:white;font-size:13px;padding:6px 12px}
      .btn-delete:hover{background:#a00}
      .status{margin-top:12px;font-size:14px;color:#555}
      .progress{display:none;margin-top:12px;color:#2c7;font-weight:600}
    </style></head><body>
      <h1>Local RAG Server</h1>
      <p>Your private document brain is <strong style="color:green">RUNNING</strong> 
         → <strong>RTX 2060 Super</strong> (embeddings)</p>

      <div class="box">
        <h2>Upload Documents</h2>
        <input type="file" id="file-input" multiple style="padding:12px;font-size:16px;width:100%;max-width:500px">
        <button id="upload-btn" class="btn btn-upload">Upload & Index</button>
        <div class="progress" id="progress">Uploading and indexing... Please wait</div>
        <div class="status" id="status"></div>
      </div>

      <div class="box">
        <h2>Indexed Documents</h2>
        <div id="doc-list">Loading...</div>
      </div>

      <div class="box">
        <h2>MCP Compatible</h2>
        <p>Base URL: <code>http://localhost:8004/mcp</code></p>        
      </div>

      <script>
      const statusEl = document.getElementById('status');
      const progressEl = document.getElementById('progress');
      const docListEl = document.getElementById('doc-list');

      async function loadDocuments() {
        try {
          const resp = await fetch('/api/documents');
          const data = await resp.json();
          if (!data.documents || data.documents.length === 0) {
            docListEl.innerHTML = '<em>No documents indexed yet.</em>';
            return;
          }
          let html = '<table><thead><tr><th>Filename</th><th style="width:130px">Action</th></tr></thead><tbody>';
          for (const doc of data.documents.sort()) {
            html += `<tr>
              <td><strong>${doc}</strong></td>
              <td>
                <button class="btn btn-delete" onclick="deleteDoc('${doc}')">
                  Delete
                </button>
              </td>
            </tr>`;
          }
          html += '</tbody></table>';
          docListEl.innerHTML = html;
        } catch (e) {
          docListEl.innerHTML = '<em>Error loading documents</em>';
        }
      }

      async function deleteDoc(name) {
        if (!confirm(`Permanently delete "${name}" and all its data?\n\nThis cannot be undone.`)) return;
        try {
          const r = await fetch(`/api/documents/${encodeURIComponent(name)}`, {method: 'DELETE'});
          const j = await r.json();
          if (r.ok) {
            alert(`Deleted "${name}"\nChunks removed: ${j.chunks_removed}`);
            loadDocuments();
          } else {
            alert('Error: ' + (j.detail || r.statusText));
          }
        } catch (e) {
          alert('Network error: ' + e);
        }
      }

      // === UPLOAD WITH NO PAGE RELOAD ===
      document.getElementById('upload-btn').addEventListener('click', async () => {
        const files = document.getElementById('file-input').files;
        if (files.length === 0) {
          statusEl.textContent = 'Please select at least one file';
          return;
        }

        progressEl.style.display = 'block';
        statusEl.textContent = `Uploading ${files.length} file(s)...`;

        const formData = new FormData();
        for (const file of files) {
          formData.append('file', file);
        }

        try {
          const resp = await fetch('/api/upload', {
            method: 'POST',
            body: formData
          });
          const result = await resp.json();

          if (resp.ok) {
            statusEl.innerHTML = `<span style="color:green">Success! Indexed: ${result.filename || files.length + ' files'}</span>`;
            document.getElementById('file-input').value = '';  // clear input
            loadDocuments();  // refresh list instantly
          } else {
            statusEl.innerHTML = `<span style="color:red">Error: ${result.detail || 'Upload failed'}</span>`;
          }
        } catch (e) {
          statusEl.innerHTML = `<span style="color:red">Network error: ${e}</span>`;
        } finally {
          progressEl.style.display = 'none';
        }
      });

      // Auto-load documents on start + refresh every 8s
      loadDocuments();
      setInterval(loadDocuments, 8004);
      </script>
    </body></html>
    """
TOOL_DEFINITIONS = {
    "tools": [
        {
            "name": "retrieve_context",
            "description": "Retrieve top-K relevant chunks from the FAISS index for a given query.",
            "inputSchema": {
                "title": "RetrieveContextArgs",
                "description": "Arguments for retrieve_context",
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User query to embed and search against the FAISS index."
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of top results to return."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        },
        {
            "name": "list_documents",
            "description": "List all indexed documents with filename and source_path.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        },
        {
            "name": "add_document",
            "description": "Add a document to the FAISS index from an optional source_path, or rebuild from the documents directory when omitted.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Optional absolute or relative file path to index directly without copying into ./documents."
                    }
                },
                "additionalProperties": False
            }
        }
    ]
}


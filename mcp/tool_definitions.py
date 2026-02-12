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
            "description": "List all documents",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    ]
}


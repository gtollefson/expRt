"""FastAPI server for R package assistant."""

import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    package_filter: Optional[str] = None
    file_type_filter: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query_time: float
    model_info: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    vector_store_loaded: bool


# Security
security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key (simple implementation)."""
    # In production, this should check against a secure key store
    expected_key = os.getenv("API_KEY", "dev-key-123")
    if credentials.credentials != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials


# Create FastAPI app
app = FastAPI(
    title="R Package Assistant API",
    description="Local LLM-powered assistant for R package documentation",
    version="0.1.0"
)

# CORS middleware for Shiny app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state (will be initialized on startup)
assistant_state = {
    "embedding_generator": None,
    "vector_store": None,
    "llm_model": None,
    "initialized": False
}


async def initialize_models():
    """Initialize models and vector store on startup."""
    try:
        print("Initializing R Package Assistant...")
        
        # Initialize embedding generator
        # Note: This will fail without proper dependencies
        # from embeddings.embedding_generator import EmbeddingGenerator
        # assistant_state["embedding_generator"] = EmbeddingGenerator()
        
        # Load vector store if available
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "data/vectorstore")
        if Path(vector_store_path).exists():
            # from embeddings.vector_store import VectorStore
            # assistant_state["vector_store"] = VectorStore.load(vector_store_path)
            print(f"Vector store would be loaded from {vector_store_path}")
        
        # Initialize LLM (placeholder)
        # This would load a local model like Phi-3 or Mistral
        print("LLM initialization placeholder")
        
        assistant_state["initialized"] = True
        print("✅ Assistant initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize assistant: {e}")
        assistant_state["initialized"] = False


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    await initialize_models()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if assistant_state["initialized"] else "initializing",
        version="0.1.0",
        models_loaded=assistant_state["embedding_generator"] is not None,
        vector_store_loaded=assistant_state["vector_store"] is not None
    )


@app.post("/query", response_model=QueryResponse)
async def query_assistant(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """Main query endpoint for R package assistance."""
    if not assistant_state["initialized"]:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    import time
    start_time = time.time()
    
    try:
        # Mock response for now (until models are loaded)
        mock_answer = f"""Based on your query "{request.query}", here's what I found:

This is a mock response from the R Package Assistant. In the full implementation, this would:

1. Generate an embedding for your query
2. Search the vector store for relevant R documentation chunks
3. Use a local LLM to generate a comprehensive answer
4. Return the answer along with source citations

Query parameters:
- Max results: {request.max_results}
- Package filter: {request.package_filter or 'None'}
- File type filter: {request.file_type_filter or 'None'}

The system would search through R package documentation, help files, and source code to provide accurate, version-aware assistance."""

        mock_sources = [
            {
                "chunk_id": "example_1",
                "source_file": "data/test_package/R/functions.R",
                "package_name": "TestPackage",
                "score": 0.85,
                "section_header": "Function Documentation",
                "text_preview": "Example R function with roxygen documentation..."
            },
            {
                "chunk_id": "example_2", 
                "source_file": "data/test_package/DESCRIPTION",
                "package_name": "TestPackage",
                "score": 0.72,
                "section_header": "Package Metadata",
                "text_preview": "Package: TestPackage, Version: 0.1.0..."
            }
        ]
        
        query_time = time.time() - start_time
        
        return QueryResponse(
            answer=mock_answer,
            sources=mock_sources,
            query_time=query_time,
            model_info={
                "embedding_model": "all-MiniLM-L6-v2 (mock)",
                "llm_model": "local-model (mock)",
                "vector_store": "FAISS (mock)"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/stats")
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Get statistics about the vector store and models."""
    if not assistant_state["initialized"]:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    # Mock stats
    return {
        "vector_store": {
            "total_chunks": 1250,
            "packages": {"TestPackage": 45, "ggplot2": 320, "dplyr": 285},
            "file_types": {".R": 680, ".Rd": 420, ".Rmd": 150}
        },
        "models": {
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "llm_model": "microsoft/Phi-3-mini-4k-instruct"
        },
        "performance": {
            "avg_query_time": 0.45,
            "total_queries": 1337
        }
    }


# Development server
if __name__ == "__main__":
    # Set development API key
    os.environ.setdefault("API_KEY", "dev-key-123")
    
    print("🚀 Starting R Package Assistant API Server")
    print("API Key for development: dev-key-123")
    print("Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
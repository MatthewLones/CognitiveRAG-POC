"""
FastAPI endpoints for Cognitive RAG POC
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yaml
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

from rag.chain import RAGChain
from rag.ingest import DocumentIngester

app = FastAPI(title="Cognitive RAG POC", version="1.0.0")

# Global RAG chain instance (will be initialized on startup)
rag_chain = None

class QueryRequest(BaseModel):
    query: str
    config: str = "naive_plus"
    max_results: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: List[dict]
    confidence: float
    groundedness: str
    metadata: dict

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_chain
    try:
        print("Initializing RAG system...")
        
        # Initialize document ingester
        ingester = DocumentIngester()
        
        # Load and process documents
        print("Loading documents...")
        documents = ingester.load_documents()
        
        if not documents:
            print("No documents found in data directory. Please add PDF files to data/ folder.")
            return
        
        # Chunk documents
        print("Chunking documents...")
        chunks = ingester.chunk_documents(documents)
        
        # Create embeddings
        print("Creating embeddings...")
        chunks_with_embeddings = ingester.create_embeddings(chunks)
        
        # Initialize RAG chain
        print("Building RAG chain...")
        rag_chain = RAGChain()
        rag_chain.build_index(chunks_with_embeddings)
        
        print(f"RAG system initialized successfully with {len(chunks_with_embeddings)} chunks")
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        rag_chain = None

@app.get("/")
async def root():
    return {"message": "Cognitive RAG POC API", "status": "running"}

@app.get("/configs")
async def list_configs():
    """List available configurations"""
    config_dir = "configs"
    configs = []
    for file in os.listdir(config_dir):
        if file.endswith(".yaml"):
            configs.append(file.replace(".yaml", ""))
    return {"configs": configs}

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using RAG system
    """
    global rag_chain
    
    if rag_chain is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not initialized. Check server logs for details."
        )
    
    try:
        # Load configuration based on request
        config_path = f"configs/{request.config}.yaml"
        if not os.path.exists(config_path):
            config_path = "configs/naive_plus.yaml"
        
        # Create a new RAG chain instance with the requested config
        query_rag_chain = RAGChain(config_path)
        
        # Copy the index from the global instance
        query_rag_chain.retriever = rag_chain.retriever
        query_rag_chain.retriever.chunks = rag_chain.retriever.chunks
        query_rag_chain.retriever.dense_index = rag_chain.retriever.dense_index
        query_rag_chain.retriever.bm25_index = rag_chain.retriever.bm25_index
        
        # Process the query
        result = query_rag_chain.query(request.query)
        
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            confidence=result["confidence"],
            groundedness=result["groundedness"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    global rag_chain
    status = "healthy" if rag_chain is not None else "initializing"
    return {"status": status, "rag_initialized": rag_chain is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

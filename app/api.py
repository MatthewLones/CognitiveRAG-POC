"""
FastAPI endpoints for Cognitive RAG POC
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import yaml
import os

app = FastAPI(title="Cognitive RAG POC", version="1.0.0")

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
    try:
        # TODO: Implement RAG pipeline
        # This is a placeholder response
        return QueryResponse(
            answer="This is a placeholder response. RAG pipeline will be implemented.",
            citations=[],
            confidence=0.0,
            groundedness="ungrounded",
            metadata={"config": request.config, "query": request.query}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

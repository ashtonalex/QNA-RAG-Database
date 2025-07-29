"""
FastAPI endpoints for the integrated RAG pipeline.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from backend.app.services.enhanced_rag_pipeline import EnhancedRAGPipeline, PipelineConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])

# Initialize pipeline with configuration
config = PipelineConfig(
    max_candidates=20,
    rerank_top_n=10,
    timeout_seconds=30,
    max_retries=3,
    fallback_on_rerank_failure=True
)
pipeline = EnhancedRAGPipeline(config)

class QueryRequest(BaseModel):
    query: str
    collection_name: str = "default_collection"
    k: int = 20
    metadata_filter: Optional[Dict[str, Any]] = None
    rag_prompt_template: str = "Context: {context}\n\nQuestion: {query}\n\nAnswer:"

class QueryResponse(BaseModel):
    response: str
    sources: list
    pipeline_info: Dict[str, Any]

@router.post("/query", response_model=QueryResponse, summary="Process RAG query")
async def process_rag_query(request: QueryRequest):
    """
    Process a query through the complete RAG pipeline.
    
    Returns enhanced response with source citations and pipeline metadata.
    """
    try:
        result = await pipeline.process_query(
            query=request.query,
            collection_name=request.collection_name,
            k=request.k,
            metadata_filter=request.metadata_filter,
            rag_prompt_template=request.rag_prompt_template
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"RAG query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.websocket("/query/stream")
async def stream_rag_query(websocket: WebSocket):
    """
    Process RAG query with streaming response via WebSocket.
    
    Expected message format:
    {
        "query": "Your question here",
        "collection_name": "optional_collection",
        "k": 20,
        "metadata_filter": {},
        "rag_prompt_template": "optional template"
    }
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            
            # Validate required fields
            if "query" not in data:
                await websocket.send_json({
                    "error": "Missing required field: query"
                })
                continue
            
            # Set defaults
            query = data["query"]
            collection_name = data.get("collection_name", "default_collection")
            k = data.get("k", 20)
            metadata_filter = data.get("metadata_filter")
            rag_prompt_template = data.get("rag_prompt_template", "Context: {context}\n\nQuestion: {query}\n\nAnswer:")
            
            # Process query with streaming
            result = await pipeline.process_query(
                query=query,
                collection_name=collection_name,
                k=k,
                metadata_filter=metadata_filter,
                stream=True,
                websocket=websocket,
                rag_prompt_template=rag_prompt_template
            )
            
            # Send final metadata after streaming is complete
            await websocket.send_json({
                "type": "metadata",
                "sources": result["sources"],
                "pipeline_info": result["pipeline_info"]
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "error": f"Processing failed: {str(e)}"
            })
        except:
            pass  # Connection might be closed

@router.get("/health", summary="Check pipeline health")
async def check_pipeline_health():
    """
    Check the health status of all pipeline components.
    """
    try:
        health = await pipeline.health_check()
        
        # Determine overall status
        overall_status = "healthy"
        for component, status in health.items():
            if status != "ok":
                overall_status = "degraded"
                break
        
        return {
            "status": overall_status,
            "components": health,
            "message": "All components operational" if overall_status == "healthy" else "Some components have issues"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Health check failed"
        }
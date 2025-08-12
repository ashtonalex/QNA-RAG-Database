"""
API router for document processing endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List
from app.models.document_models import DocumentMetadata, DocumentStatus
from app.services.document_processor import DocumentProcessor, logger
from app.monitoring import monitor

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", summary="Upload and process a document")
@monitor.memory_check
async def upload_document(request: Request, file: UploadFile = File(...)):
    processor = DocumentProcessor()
    user_ip = request.client.host if request.client else "unknown"
    try:
        doc_id = await processor.handle_upload(file)
        monitor.log_memory_usage(f"upload_complete_{doc_id}")
        logger.info(
            f"AUDIT: User IP {user_ip} uploaded file '{file.filename}' as document ID {doc_id}"
        )
        return {"document_id": doc_id}
    except HTTPException as e:
        logger.warning(
            f"AUDIT: Upload failed from IP {user_ip} for file '{file.filename}': {e.detail}"
        )
        raise
    except Exception as e:
        logger.error(
            f"AUDIT: Unexpected error from IP {user_ip} for file '{file.filename}': {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/", response_model=List[DocumentMetadata], summary="List processed documents"
)
async def list_documents(request: Request):
    processor = DocumentProcessor()
    
    # Get all document IDs and return their metadata
    doc_ids = list(processor.redis.smembers('doc_ids'))
    documents = []
    
    for doc_id in doc_ids:
        doc_meta = processor.redis.hgetall(f"doc_meta:{doc_id}")
        if doc_meta:
            documents.append(DocumentMetadata(**doc_meta))
    
    return documents


@router.get("/{id}", response_model=DocumentMetadata, summary="Get document details")
async def get_document(id: str):
    processor = DocumentProcessor()
    meta = processor.get_document_metadata(id)
    if not meta:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentMetadata(**meta)


@router.delete("/{id}", summary="Remove document")
async def delete_document(id: str):
    processor = DocumentProcessor()
    meta = processor.get_document_metadata(id)
    if not meta:
        raise HTTPException(status_code=404, detail="Document not found")
    processor.delete_document(id)
    return {"detail": "Document deleted"}


@router.get(
    "/{id}/status", response_model=DocumentStatus, summary="Get processing status"
)
async def get_document_status(id: str):
    processor = DocumentProcessor()
    progress = processor.get_progress(id)
    if not progress:
        raise HTTPException(
            status_code=404, detail="Document not found or no progress available"
        )
    return DocumentStatus(**progress)


@router.post("/cleanup", summary="Clean up orphaned document data")
async def cleanup_documents():
    """Force cleanup of orphaned document data"""
    processor = DocumentProcessor()
    
    # Get all document IDs
    doc_ids = list(processor.redis.smembers('doc_ids'))
    cleaned_count = 0
    
    for doc_id in doc_ids:
        # Check if document has both metadata and chunks
        has_meta = processor.redis.exists(f"doc_meta:{doc_id}")
        has_chunks = processor.redis.exists(f"doc_chunks:{doc_id}")
        
        # If missing either, clean up completely
        if not has_meta or not has_chunks:
            processor.redis.delete(f"doc_meta:{doc_id}")
            processor.redis.delete(f"doc_progress:{doc_id}")
            processor.redis.delete(f"doc_chunks:{doc_id}")
            processor.redis.srem("doc_ids", doc_id)
            cleaned_count += 1
            logger.info(f"Cleaned up orphaned document: {doc_id}")
    
    remaining_docs = len(list(processor.redis.smembers('doc_ids')))
    return {"message": f"Cleaned up {cleaned_count} orphaned documents", "remaining_docs": remaining_docs}

@router.post("/reset", summary="DANGER: Remove ALL documents from Redis")
async def reset_all_documents():
    """DANGER: Remove ALL documents from Redis - use for testing only"""
    processor = DocumentProcessor()
    
    # Get all document IDs
    doc_ids = list(processor.redis.smembers('doc_ids'))
    
    # Remove all document-related data
    for doc_id in doc_ids:
        processor.redis.delete(f"doc_meta:{doc_id}")
        processor.redis.delete(f"doc_progress:{doc_id}")
        processor.redis.delete(f"doc_chunks:{doc_id}")
    
    # Clear the document IDs set
    processor.redis.delete('doc_ids')
    
    logger.info(f"Reset complete: removed {len(doc_ids)} documents")
    return {"message": f"Reset complete: removed {len(doc_ids)} documents", "remaining_docs": 0}

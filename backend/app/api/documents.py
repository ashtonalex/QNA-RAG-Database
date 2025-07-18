"""
API router for document processing endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List
from app.models.document_models import DocumentMetadata, DocumentStatus
from app.services.document_processor import DocumentProcessor, logger

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", summary="Upload and process a document")
async def upload_document(request: Request, file: UploadFile = File(...)):
    processor = DocumentProcessor()
    user_ip = request.client.host if request.client else "unknown"
    try:
        doc_id = await processor.handle_upload(file)
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
async def list_documents():
    processor = DocumentProcessor()
    docs = processor.list_documents()
    return [DocumentMetadata(**doc) for doc in docs]


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

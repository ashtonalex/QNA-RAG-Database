"""
API router for document processing endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.models.document_models import DocumentMetadata, DocumentStatus

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", summary="Upload and process a document")
async def upload_document(file: UploadFile = File(...)):
    # Handle file upload, validation, and background processing
    # Return task ID or document ID
    pass


@router.get(
    "/", response_model=List[DocumentMetadata], summary="List processed documents"
)
async def list_documents():
    # Return list of processed documents
    pass


@router.get("/{id}", response_model=DocumentMetadata, summary="Get document details")
async def get_document(id: str):
    # Return document metadata/details
    pass


@router.delete("/{id}", summary="Remove document")
async def delete_document(id: str):
    # Delete document and associated data
    pass


@router.get(
    "/{id}/status", response_model=DocumentStatus, summary="Get processing status"
)
async def get_document_status(id: str):
    # Return processing status (pending, processing, done, error)
    pass

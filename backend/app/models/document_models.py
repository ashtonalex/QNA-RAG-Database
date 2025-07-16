"""
Pydantic models for document metadata, status, and upload.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DocumentMetadata(BaseModel):
    id: str
    filename: str
    filetype: str
    size: int
    created_at: datetime
    author: Optional[str]
    word_count: Optional[int]
    language: Optional[str]
    headings: Optional[List[str]]
    tables: Optional[int]
    hash: str


class DocumentStatus(BaseModel):
    id: str
    status: str = Field(
        ..., description="Processing status: pending, processing, done, error"
    )
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    error_message: Optional[str] = None


class DocumentUploadRequest(BaseModel):
    filename: str
    filetype: str
    size: int

"""
Service for document processing: upload handling, format detection, routing, validation, and progress tracking.
"""

import os
import uuid
import magic
import hashlib
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException
from pathlib import Path
import aiofiles
import logging
import redis
import inspect
from backend.app.models.chunk_models import ChunkingConfig
import mimetypes
import docx
from pypdf import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File size limits (in bytes)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MIN_FILE_SIZE = 1024  # 1KB

# Supported file types and their magic numbers
SUPPORTED_TYPES = {
    "application/pdf": [b"%PDF"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
        b"PK\x03\x04"
    ],  # DOCX
    "text/plain": [
        b"\xef\xbb\xbf",
        b"\xff\xfe",
        b"\xfe\xff",
    ],  # UTF-8 BOM, UTF-16 LE, UTF-16 BE
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/tiff": [b"II*\x00", b"MM\x00"],
    "image/bmp": [b"BM"],
    "image/gif": [b"GIF87a", b"GIF89a"],
}

# File extensions mapping
EXTENSION_MAPPING = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}


class DocumentProcessor:
    # Mapping from MIME type to ChunkingConfig
    CHUNKING_CONFIGS = {
        "application/pdf": ChunkingConfig(
            chunk_size=500, overlap=0.1, strategy="hybrid"
        ),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ChunkingConfig(
            chunk_size=400, overlap=0.15, strategy="hybrid"
        ),
        "text/plain": ChunkingConfig(chunk_size=300, overlap=0.2, strategy="syntactic"),
        # To add a new document type, add a new MIME type and ChunkingConfig here.
    }

    @classmethod
    def get_chunking_config_for_type(cls, filetype: str) -> ChunkingConfig:
        """
        Return the ChunkingConfig for a given MIME type, or a default config if not found.
        Extend CHUNKING_CONFIGS to support new document types.
        """
        return cls.CHUNKING_CONFIGS.get(filetype, ChunkingConfig())

    def __init__(self):
        """
        DocumentProcessor handles file upload, validation, text extraction, chunking, and storage.
        Uses per-document-type chunking configuration for flexible processing.
        """
        # Initialize storage paths
        self.upload_dir = Path("temp_uploads")
        self.upload_dir.mkdir(exist_ok=True)

        # Initialize magic number detector
        self.magic_detector = magic.Magic(mime=True)

        # Initialize OCR service
        from .ocr_service import OCRService

        self.ocr_service = OCRService()

        # Initialize Redis connection
        self.redis = redis.Redis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )

        logger.info("DocumentProcessor initialized")

    async def _extract_text_from_file(self, filepath: str, filetype: str) -> str:
        """
        Extract text from a file based on its MIME type.
        Supports PDF, DOCX, and TXT. Extend this method to support more types.
        """
        if filetype == "application/pdf":
            try:
                reader = PdfReader(filepath)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
                return text
            except Exception as e:
                logger.error(f"PDF extraction failed: {e}")
                return ""
        elif (
            filetype
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            try:
                doc = docx.Document(filepath)
                text = "\n".join([para.text for para in doc.paragraphs])
                return text
            except Exception as e:
                logger.error(f"DOCX extraction failed: {e}")
                return ""
        elif filetype == "text/plain":
            try:
                async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
                    return await f.read()
            except Exception as e:
                logger.error(f"TXT extraction failed: {e}")
                return ""
        else:
            logger.warning(f"Unsupported file type for extraction: {filetype}")
            return ""

    async def handle_upload(self, file: UploadFile) -> str:
        """
        Handle async file upload, validation, text extraction, chunking, and chunk storage.
        Pipeline:
        1. Validate and save file
        2. Select chunking config based on file type
        3. Extract text from file
        4. Chunk text using async chunking service
        5. Store resulting chunks in Redis
        Returns document ID.
        """
        try:
            # Generate unique document ID
            doc_id = str(uuid.uuid4())
            logger.info(f"Processing upload for document ID: {doc_id}")
            # Track progress: upload started
            self.track_progress(doc_id, status="pending", progress=0)

            # Validate file size
            await self._validate_file_size(file)

            # Read file content for validation
            content = await file.read()

            # Validate file type using magic numbers
            if not file.filename:
                self.track_progress(
                    doc_id, status="error", error_message="Filename is required"
                )
                raise HTTPException(status_code=400, detail="Filename is required")
            filename = file.filename
            detected_type = await self._validate_file_type(content, filename)

            # Select chunking config for this document type
            chunking_config = self.get_chunking_config_for_type(detected_type)
            logger.info(
                f"Selected chunking config for {detected_type}: {chunking_config}"
            )

            # (Stub) Malware check
            await self._check_malware(content, doc_id)

            # Save file to temporary storage
            filepath = await self._save_file(content, filename, doc_id)
            # Track progress: file saved
            self.track_progress(doc_id, status="processing", progress=25)

            # Calculate file hash
            file_hash = hashlib.sha256(content).hexdigest()

            # Store metadata in Redis
            import datetime

            metadata = {
                "id": doc_id,
                "filename": filename,
                "filetype": detected_type,
                "size": len(content),
                "created_at": datetime.datetime.utcnow().isoformat(),
                "author": None,
                "word_count": None,
                "language": None,
                "headings": None,
                "tables": None,
                "hash": file_hash,
            }
            # Convert None values to empty strings for Redis compatibility
            metadata = {k: ("" if v is None else v) for k, v in metadata.items()}
            self.redis.hmset(f"doc_meta:{doc_id}", metadata)
            self.redis.sadd("doc_ids", doc_id)

            logger.info(
                f"File uploaded successfully: {doc_id}, type: {detected_type}, size: {len(content)} bytes"
            )

            # Extract text from file
            extracted_text = await self._extract_text_from_file(filepath, detected_type)
            logger.info(f"Extracted text length: {len(extracted_text)}")

            # Chunk the extracted text
            from backend.app.services.chunking_service import ChunkingService

            chunker = ChunkingService(chunking_config)
            chunks = await chunker.hybrid_chunk(extracted_text, metadata={})
            logger.info(
                f"Chunked into {len(chunks)} chunks. Sample: {chunks[0].text if chunks else 'No chunks'}"
            )

            # Store chunks in Redis
            self._store_chunks(doc_id, chunks)

            # Track progress: extraction done
            self.track_progress(doc_id, status="processing", progress=75)

            # Simulate extraction step (replace with actual extraction logic as needed)
            try:
                # Extraction (e.g., OCR or text extraction)
                # For demonstration, just call extract_text if you want real extraction
                # extracted_text = self.extract_text(filepath)
                # Track progress: extraction done
                self.track_progress(doc_id, status="processing", progress=75)
            except Exception as e:
                self.track_progress(doc_id, status="error", error_message=str(e))
                raise

            # Track progress: done
            self.track_progress(doc_id, status="done", progress=100)
            return doc_id

        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            # Track progress: error
            if "doc_id" in locals():
                self.track_progress(doc_id, status="error", error_message=str(e))
            raise HTTPException(status_code=400, detail=str(e))

    async def _validate_file_size(self, file: UploadFile) -> None:
        """
        Validate file size limits.
        """
        # Get file size by reading content (for async compatibility)
        content = await file.read()
        file_size = len(content)

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB",
            )

        if file_size < MIN_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too small. Minimum size is {MIN_FILE_SIZE} bytes",
            )

        # Reset file position for further processing
        await file.seek(0)

    async def _validate_file_type(self, content: bytes, filename: str) -> str:
        """
        Validate file type using magic numbers.
        """
        # Get file extension
        file_ext = Path(filename).suffix.lower()

        # Check if extension is supported
        if file_ext not in EXTENSION_MAPPING:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file extension: {file_ext}"
            )

        # Detect MIME type using magic numbers
        detected_mime = self.magic_detector.from_buffer(content)

        # Get expected MIME type from extension
        expected_mime = EXTENSION_MAPPING[file_ext]

        # Validate magic numbers
        if not self._check_magic_numbers(content, expected_mime):
            raise HTTPException(
                status_code=400,
                detail=f"File content doesn't match expected type. Expected: {expected_mime}, Detected: {detected_mime}",
            )

        # Additional validation: check if detected MIME matches expected
        if detected_mime != expected_mime:
            logger.warning(
                f"MIME type mismatch for {filename}: expected {expected_mime}, detected {detected_mime}"
            )
            # For some files, we might want to be more lenient, but for now we'll be strict
            if detected_mime not in SUPPORTED_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type detected: {detected_mime}",
                )

        return detected_mime

    def _check_magic_numbers(self, content: bytes, expected_mime: str) -> bool:
        """
        Check if file content matches expected magic numbers.
        """
        if expected_mime not in SUPPORTED_TYPES:
            return False

        expected_magic_numbers = SUPPORTED_TYPES[expected_mime]

        for magic_number in expected_magic_numbers:
            if content.startswith(magic_number):
                return True

        return False

    async def _check_malware(self, content: bytes, doc_id: str) -> None:
        """
        (Stub) Malware check - placeholder for future implementation.
        In a real implementation, this would integrate with antivirus services.
        For now, implementation - always passes
        # TODO: Integrate with antivirus service (e.g., ClamAV, VirusTotal API)
        """
        logger.info(f"Malware check stub for document {doc_id}")
        # Example of what a real implementation might look like:
        # - Check file hash against known malware databases
        # - Scan with antivirus engine
        # - Check file behavior patterns
        # - Validate file structure integrity
        # For now, we'll do some basic sanity checks
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        # Check for suspicious patterns (very basic)
        suspicious_patterns = [
            b"\x4d\x5a",  # MZ header (Windows executable)
            b"\x7fELF",  # ELF header (Linux executable)
        ]
        for pattern in suspicious_patterns:
            if content.startswith(pattern):
                logger.warning(f"Suspicious file pattern detected in {doc_id}")
                # In a real implementation, you might want to block these
                # For now, we'll just log a warning

    async def _save_file(self, content: bytes, filename: str, doc_id: str) -> str:
        """
        Save file to temporary storage.
        """
        # Create safe filename
        safe_filename = self._sanitize_filename(filename)
        filepath = self.upload_dir / f"{doc_id}_{safe_filename}"
        # Save file asynchronously
        async with aiofiles.open(filepath, "wb") as f:
            await f.write(content)
        logger.info(f"File saved to: {filepath}")
        return str(filepath)

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and other security issues.
        """
        # Remove path separators and other dangerous characters
        dangerous_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        sanitized = filename
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "_")
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255] + ext
        return sanitized

    def track_progress(
        self,
        doc_id: str,
        status: str,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
    ):
        """
        Store or update document processing progress in Redis.
        """
        progress_data = {
            "id": doc_id,
            "status": status,
        }
        if progress is not None:
            progress_data["progress"] = str(progress)
        if error_message:
            progress_data["error_message"] = error_message
        self.redis.hmset(f"doc_progress:{doc_id}", progress_data)

    def get_progress(self, doc_id: str):
        """
        Retrieve document processing progress from Redis.
        """
        data_or_awaitable = self.redis.hgetall(f"doc_progress:{doc_id}")
        if inspect.isawaitable(data_or_awaitable):
            import asyncio

            data = asyncio.get_event_loop().run_until_complete(data_or_awaitable)
        else:
            data = data_or_awaitable
        if not data:
            return None
        # Convert progress to float if present
        if "progress" in data:
            data["progress"] = float(data["progress"])
        return data

    def list_documents(self):
        """
        List all document metadata.
        """
        doc_ids_or_awaitable = self.redis.smembers("doc_ids")
        if inspect.isawaitable(doc_ids_or_awaitable):
            import asyncio

            doc_ids = asyncio.get_event_loop().run_until_complete(doc_ids_or_awaitable)
        else:
            doc_ids = doc_ids_or_awaitable
        docs = []
        for doc_id in doc_ids:
            meta_or_awaitable = self.redis.hgetall(f"doc_meta:{doc_id}")
            if inspect.isawaitable(meta_or_awaitable):
                import asyncio

                meta = asyncio.get_event_loop().run_until_complete(meta_or_awaitable)
            else:
                meta = meta_or_awaitable
            if meta:
                # Convert fields to correct types
                if "size" in meta:
                    meta["size"] = int(meta["size"])
                if "created_at" in meta:
                    from datetime import datetime

                    meta["created_at"] = datetime.fromisoformat(meta["created_at"])
                docs.append(meta)
        return docs

    def get_document_metadata(self, doc_id: str):
        """
        Get metadata for a specific document.
        """
        meta_or_awaitable = self.redis.hgetall(f"doc_meta:{doc_id}")
        if inspect.isawaitable(meta_or_awaitable):
            import asyncio

            meta = asyncio.get_event_loop().run_until_complete(meta_or_awaitable)
        else:
            meta = meta_or_awaitable
        if not meta:
            return None
        if "size" in meta:
            meta["size"] = int(meta["size"])
        if "created_at" in meta:
            from datetime import datetime

            meta["created_at"] = datetime.fromisoformat(meta["created_at"])
        return meta

    def delete_document(self, doc_id: str):
        """
        Delete document file, metadata, and progress.
        """
        # Remove file
        meta_or_awaitable = self.redis.hgetall(f"doc_meta:{doc_id}")
        if inspect.isawaitable(meta_or_awaitable):
            import asyncio

            meta = asyncio.get_event_loop().run_until_complete(meta_or_awaitable)
        else:
            meta = meta_or_awaitable
        if meta and "filename" in meta:
            safe_filename = self._sanitize_filename(meta["filename"])
            filepath = self.upload_dir / f"{doc_id}_{safe_filename}"
            if filepath.exists():
                filepath.unlink()
        # Remove metadata and progress
        self.redis.delete(f"doc_meta:{doc_id}")
        self.redis.delete(f"doc_progress:{doc_id}")
        self.redis.srem("doc_ids", doc_id)

    def _store_chunks(self, doc_id: str, chunks: list) -> None:
        """
        Store the list of chunk dicts in Redis under 'doc_chunks:{doc_id}'.
        Chunks are serialized as strings for persistence.
        """
        chunk_dicts = [chunk.dict() for chunk in chunks]
        self.redis.delete(f"doc_chunks:{doc_id}")  # Remove any existing
        if chunk_dicts:
            self.redis.rpush(f"doc_chunks:{doc_id}", *[str(cd) for cd in chunk_dicts])
        logger.info(f"Stored {len(chunk_dicts)} chunks for doc {doc_id}")

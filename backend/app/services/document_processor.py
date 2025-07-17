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
    def __init__(self):
        # Initialize storage paths
        self.upload_dir = Path("temp_uploads")
        self.upload_dir.mkdir(exist_ok=True)

        # Initialize magic number detector
        self.magic_detector = magic.Magic(mime=True)

        # Initialize OCR service
        from backend.app.services.ocr_service import OCRService

        self.ocr_service = OCRService()

        logger.info("DocumentProcessor initialized")

    async def handle_upload(self, file: UploadFile) -> str:
        """
        Handle async file upload, validate, and save to temp storage.
        Returns a unique document/task ID.
        """
        try:
            # Generate unique document ID
            doc_id = str(uuid.uuid4())
            logger.info(f"Processing upload for document ID: {doc_id}")

            # Validate file size
            await self._validate_file_size(file)

            # Read file content for validation
            content = await file.read()

            # Validate file type using magic numbers
            if not file.filename:
                raise HTTPException(status_code=400, detail="Filename is required")
            filename = file.filename
            detected_type = await self._validate_file_type(content, filename)

            # (Stub) Malware check
            await self._check_malware(content, doc_id)

            # Save file to temporary storage
            filepath = await self._save_file(content, filename, doc_id)

            # Calculate file hash
            file_hash = hashlib.sha256(content).hexdigest()

            logger.info(
                f"File uploaded successfully: {doc_id}, type: {detected_type}, size: {len(content)} bytes"
            )

            return doc_id

        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
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

    def detect_format(self, filepath: str) -> str:
        """
        Detect the file format using python-magic and route to the correct extraction method.
        Returns the detected MIME type.
        """
        if not os.path.exists(filepath):
            raise HTTPException(status_code=400, detail=f"File not found: {filepath}")

        with open(filepath, "rb") as f:
            header = f.read(2048)
            mime_type = self.magic_detector.from_buffer(header)

        if mime_type == "application/pdf":
            return "pdf"
        elif (
            mime_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            return "docx"
        elif mime_type == "text/plain":
            return "txt"
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {mime_type}"
            )

    def extract_text(self, filepath: str) -> str:
        """
        Detect format and extract text using the appropriate method.
        """
        filetype = self.detect_format(filepath)
        if filetype == "pdf":
            return self._extract_pdf(filepath)
        elif filetype == "docx":
            return self._extract_docx(filepath)
        elif filetype == "txt":
            return self._extract_txt(filepath)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for extraction: {filetype}",
            )

    def _extract_pdf(self, filepath: str) -> str:
        """
        Extract text from a PDF file using pypdf or OCR if image-based.
        """
        from pypdf import PdfReader

        try:
            # Check if image-based PDF
            if self.ocr_service.is_image_pdf(filepath):
                # Convert PDF pages to images
                images = self.ocr_service.pdf_to_images(filepath)
                text = ""
                for image in images:
                    page_text, _ = self.ocr_service.run_ocr(image)
                    text += page_text + "\n"
                return text
            else:
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to extract text from PDF."
            )

    def _extract_docx(self, filepath: str) -> str:
        """
        Extract text from a DOCX file using python-docx.
        """
        import docx

        try:
            doc = docx.Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to extract text from DOCX."
            )

    def _extract_txt(self, filepath: str) -> str:
        """
        Extract text from a TXT file.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to extract text from TXT."
            )

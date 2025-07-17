import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from backend.app.models.chunk_models import ChunkingConfig
from backend.app.services.document_processor import DocumentProcessor
from backend.app.services.chunking_service import ChunkingService


@pytest.fixture
def processor():
    return DocumentProcessor()


@pytest.mark.asyncio
async def test_per_document_type_config_selection():
    pdf_config = DocumentProcessor.get_chunking_config_for_type("application/pdf")
    assert isinstance(pdf_config, ChunkingConfig)
    assert pdf_config.chunk_size == 500
    txt_config = DocumentProcessor.get_chunking_config_for_type("text/plain")
    assert txt_config.strategy == "syntactic"
    unknown_config = DocumentProcessor.get_chunking_config_for_type("unknown/type")
    assert isinstance(unknown_config, ChunkingConfig)


@pytest.mark.asyncio
async def test_async_text_extraction_and_chunking(tmp_path):
    # Create a sample TXT file
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello world!\nThis is a test.\nAnother line.")
    processor = DocumentProcessor()
    config = DocumentProcessor.get_chunking_config_for_type("text/plain")
    text = await processor._extract_text_from_file(str(txt_file), "text/plain")
    assert "Hello world!" in text
    chunker = ChunkingService(config)
    chunks = await chunker.hybrid_chunk(text, metadata={})
    assert len(chunks) > 0
    assert any("Hello world!" in c.text for c in chunks)


@pytest.mark.asyncio
async def test_chunk_storage(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("A. B. C. D. E. F. G.")
    processor = DocumentProcessor()
    config = DocumentProcessor.get_chunking_config_for_type("text/plain")
    text = await processor._extract_text_from_file(str(txt_file), "text/plain")
    chunker = ChunkingService(config)
    chunks = await chunker.hybrid_chunk(text, metadata={})
    doc_id = "test-doc-id"
    processor._store_chunks(doc_id, chunks)
    # Check Redis for stored chunks
    import inspect

    stored = processor.redis.lrange(f"doc_chunks:{doc_id}", 0, -1)
    if inspect.isawaitable(stored):
        stored = await stored
    # Now stored is always a list
    assert len(stored) == len(chunks)
    assert any("A." in s for s in stored)

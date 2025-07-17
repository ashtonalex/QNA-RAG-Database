import pytest
import asyncio
from backend.app.models.chunk_models import ChunkingConfig
from backend.app.services.chunking_service import chunk_document, ChunkingService


@pytest.mark.asyncio
async def test_chunk_document_and_vector_payloads():
    text = (
        "This is the first sentence. Here is another one! Yet another sentence follows."
    )
    config = ChunkingConfig(chunk_size=10, overlap=0.1, strategy="hybrid")
    chunks = await chunk_document(text, config)
    assert len(chunks) > 0
    # Test output format for vector DB
    payloads = ChunkingService(config).chunks_to_vector_payloads(chunks)
    assert isinstance(payloads, list)
    assert all("id" in p and "text" in p and "metadata" in p for p in payloads)
    # Test with a different config
    config2 = ChunkingConfig(chunk_size=2, overlap=0.0, strategy="syntactic")
    chunks2 = await chunk_document(text, config2)
    payloads2 = ChunkingService(config2).chunks_to_vector_payloads(chunks2)
    assert len(payloads2) > 0
    assert all(isinstance(p["text"], str) for p in payloads2)

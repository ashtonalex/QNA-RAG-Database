"""
Integration tests for the vector pipeline, including chunking, embedding, storage, and retrieval using VectorService and ChromaDB.
"""

import pytest
import asyncio
import hashlib
import time
from backend.app.services.vector_service import VectorService

TEST_COLLECTION = "test_collection_pipeline"

@pytest.mark.asyncio
async def test_full_pipeline_correctness_and_cleanup():
    service = VectorService()
    text = (
        "Section 1. This is a test document. It contains several sentences. "
        "Section 2. Here is another section. This section has more content."
    )
    doc_id = "doc_test_001"
    section_title = "Test Section"
    page_number = 1

    # Chunking
    chunks = await service.hybrid_chunk(text, doc_id, section_title, page_number)
    assert chunks, "Chunking should produce at least one chunk"
    for chunk in chunks:
        assert "chunk" in chunk and "metadata" in chunk
        assert chunk["metadata"]["doc_id"] == doc_id

    # Embedding & Storage
    await service.store_chunks(chunks, TEST_COLLECTION)

    # Retrieval by ID (hash)
    collection = await service.create_collection(TEST_COLLECTION)
    for chunk in chunks:
        chunk_hash = hashlib.sha256(chunk["chunk"].encode()).hexdigest()
        result = collection.get(ids=[chunk_hash])
        assert result and result["documents"], "Chunk should be retrievable by hash"

    # Cleanup
    service.client.delete_collection(TEST_COLLECTION)

@pytest.mark.asyncio
async def test_pipeline_robustness_empty_input():
    service = VectorService()
    chunks = await service.hybrid_chunk("", "doc_empty", "Empty Section", 0)
    assert chunks == [], "Empty input should produce no chunks"

@pytest.mark.asyncio
async def test_pipeline_robustness_large_input():
    service = VectorService()
    text = " ".join(["Sentence."] * 5000)  # Large document
    doc_id = "doc_large"
    section_title = "Large Section"
    page_number = 99
    chunks = await service.hybrid_chunk(text, doc_id, section_title, page_number)
    assert len(chunks) > 0, "Large input should be chunked"
    await service.store_chunks(chunks, TEST_COLLECTION)
    service.client.delete_collection(TEST_COLLECTION)

@pytest.mark.asyncio
async def test_pipeline_performance():
    service = VectorService()
    text = " ".join(["Sentence."] * 2000)
    doc_id = "doc_perf"
    section_title = "Perf Section"
    page_number = 2
    start = time.time()
    chunks = await service.hybrid_chunk(text, doc_id, section_title, page_number)
    await service.store_chunks(chunks, TEST_COLLECTION)
    elapsed = time.time() - start
    service.client.delete_collection(TEST_COLLECTION)
    assert elapsed < 10, f"Pipeline should process in reasonable time, got {elapsed:.2f}s"

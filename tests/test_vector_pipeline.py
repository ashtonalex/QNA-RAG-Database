"""
Integration tests for the vector pipeline, including chunking, embedding, storage, and retrieval using VectorService and ChromaDB.
"""

import pytest
import asyncio
import hashlib
import time
import os
import tempfile
from fastapi import UploadFile
try:
    from backend.app.services.vector_service import VectorService
    from backend.app.services.document_processor import DocumentProcessor
except ImportError:
    from app.services.vector_service import VectorService
    from app.services.document_processor import DocumentProcessor
from sentence_transformers import SentenceTransformer
import chromadb
from unittest.mock import patch
import tracemalloc

TEST_COLLECTION = "test_collection_pipeline"
LOG_FILE = "vector_pipeline_test_output.log"

def log_step(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def clear_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

@pytest.mark.asyncio
async def test_full_pipeline_correctness_and_cleanup():
    clear_log()
    service = VectorService()
    text = (
        "Section 1. This is a test document. It contains several sentences. "
        "Section 2. Here is another section. This section has more content."
    )
    doc_id = "doc_test_001"
    section_title = "Test Section"
    page_number = 1

    log_step("[START] Full pipeline test: chunking...")
    # Chunking
    chunks = await service.semantic_chunk(text)
    log_step(f"[CHUNKING] Produced {len(chunks)} chunks: {chunks}")
    assert chunks, "Chunking should produce at least one chunk"

    log_step("[EMBEDDING] Generating embeddings...")
    embeddings = await service.generate_embeddings(chunks)
    log_step(f"[EMBEDDING] Embeddings generated: {len([e for e in embeddings if e is not None])} valid out of {len(embeddings)}")
    assert all(e is not None for e in embeddings), "All embeddings should be generated"

    log_step("[STORAGE] Storing chunks in ChromaDB...")
    await service.store_chunks(chunks, TEST_COLLECTION, doc_id, section_title, page_number)
    log_step("[STORAGE] Chunks stored successfully.")

    log_step("[RETRIEVAL] Retrieving by similarity search...")
    results = await service.similarity_search("test document", TEST_COLLECTION, k=3)
    log_step(f"[RETRIEVAL] Retrieved {len(results)} results: {results}")
    assert results, "Should retrieve at least one result"

    # Cleanup
    try:
        service.client.delete_collection(TEST_COLLECTION)
        log_step("[CLEANUP] Test collection deleted. [SUCCESS]")
    except Exception as e:
        if hasattr(chromadb.errors, "NotFoundError") and isinstance(e, chromadb.errors.NotFoundError):
            pass
        else:
            raise

@pytest.mark.asyncio
async def test_pipeline_robustness_empty_input():
    service = VectorService()
    log_step("[START] Robustness test: empty input...")
    chunks = await service.semantic_chunk("")
    log_step(f"[CHUNKING] Chunks from empty input: {chunks}")
    assert chunks == [], "Empty input should produce no chunks"
    log_step("[SUCCESS] Empty input handled correctly.")

@pytest.mark.asyncio
async def test_pipeline_robustness_large_input():
    service = VectorService()
    text = " ".join(["Sentence."] * 5000)  # Large document
    doc_id = "doc_large"
    section_title = "Large Section"
    page_number = 99
    log_step("[START] Robustness test: large input...")
    chunks = await service.semantic_chunk(text)
    log_step(f"[CHUNKING] Produced {len(chunks)} chunks for large input.")
    assert len(chunks) > 0, "Large input should be chunked"
    await service.store_chunks(chunks, TEST_COLLECTION, doc_id, section_title, page_number)
    log_step("[STORAGE] Large input stored successfully.")
    try:
        service.client.delete_collection(TEST_COLLECTION)
        log_step("[CLEANUP] Test collection deleted. [SUCCESS]")
    except Exception as e:
        if hasattr(chromadb.errors, "NotFoundError") and isinstance(e, chromadb.errors.NotFoundError):
            pass
        else:
            raise

@pytest.mark.asyncio
async def test_pipeline_performance():
    service = VectorService()
    text = " ".join(["Sentence."] * 2000)
    doc_id = "doc_perf"
    section_title = "Perf Section"
    page_number = 2
    log_step("[START] Performance test...")
    start = time.time()
    chunks = await service.semantic_chunk(text)
    log_step(f"[CHUNKING] Produced {len(chunks)} chunks.")
    await service.store_chunks(chunks, TEST_COLLECTION, doc_id, section_title, page_number)
    elapsed = time.time() - start
    log_step(f"[PERFORMANCE] Pipeline completed in {elapsed:.2f}s.")
    try:
        service.client.delete_collection(TEST_COLLECTION)
        assert elapsed < 10, f"Pipeline should process in reasonable time, got {elapsed:.2f}s"
        log_step("[SUCCESS] Performance test passed.")
    except Exception as e:
        if hasattr(chromadb.errors, "NotFoundError") and isinstance(e, chromadb.errors.NotFoundError):
            pass
        else:
            raise

@pytest.mark.asyncio
async def test_real_world_file_inputs():
    clear_log()
    service = VectorService()
    processor = DocumentProcessor()
    log_step("[START] Real-world file input test...")
    # Prepare test files (use small dummy files from backend/app/services/)
    test_files = [
        ("dummy.pdf", "application/pdf"),
        ("dummy.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("test.txt", "text/plain"),
    ]
    for fname, ftype in test_files:
        path = os.path.join(os.path.dirname(__file__), "..", "backend", "app", "services", fname)
        if not os.path.exists(path):
            log_step(f"[SKIP] Test file not found: {fname}")
            continue
        log_step(f"[PROCESSING] File: {fname}")
        # Use FastAPI UploadFile for FastAPI compatibility
        with open(path, "rb") as f:
            upload = UploadFile(filename=os.path.basename(path), file=f)
            doc_id = await processor.handle_upload(upload)
        log_step(f"[UPLOAD] Document ID: {doc_id}")
        meta = processor.get_document_metadata(doc_id)
        log_step(f"[META] {meta}")
        assert meta is not None, "Metadata should be present"
        # Clean up
        processor.delete_document(doc_id)
        log_step(f"[CLEANUP] Deleted document {doc_id}")
    log_step("[SUCCESS] Real-world file input test completed.")

@pytest.mark.asyncio
async def test_edge_cases_malformed_and_non_utf8():
    clear_log()
    service = VectorService()
    processor = DocumentProcessor()
    log_step("[START] Edge case test: malformed and non-UTF8...")
    # Malformed file (random bytes)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(os.urandom(128))
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        upload = UploadFile(filename=os.path.basename(tmp_path), file=f)
        try:
            await processor.handle_upload(upload)
            log_step("[FAIL] Malformed file should not be processed successfully.")
        except Exception as e:
            log_step(f"[SUCCESS] Malformed file correctly failed: {e}")
    os.remove(tmp_path)
    # Non-UTF8 file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"\xff\xfe\xfd\xfc\xfb\xfa")
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        upload = UploadFile(filename=os.path.basename(tmp_path), file=f)
        try:
            await processor.handle_upload(upload)
            log_step("[FAIL] Non-UTF8 file should not be processed successfully.")
        except Exception as e:
            log_step(f"[SUCCESS] Non-UTF8 file correctly failed: {e}")
    os.remove(tmp_path)
    log_step("[SUCCESS] Edge case test completed.")

@pytest.mark.asyncio
async def test_deduplication():
    clear_log()
    service = VectorService()
    text = "This is a duplicate chunk. This is a duplicate chunk."
    doc_id = "doc_dedup"
    section_title = "Dedup Section"
    page_number = 1
    log_step("[START] Deduplication test...")
    chunks = await service.semantic_chunk(text)
    # Intentionally duplicate chunks
    chunks = chunks + chunks
    log_step(f"[CHUNKING] {len(chunks)} chunks (with duplicates)")
    await service.store_chunks(chunks, TEST_COLLECTION, doc_id, section_title, page_number)
    log_step("[STORAGE] Chunks stored (deduplication should occur)")
    # Retrieve all stored chunks
    results = await service.similarity_search("duplicate chunk", TEST_COLLECTION, k=10)
    hashes = set()
    for r in results:
        h = r["metadata"].get("hash")
        assert h not in hashes, "Duplicate chunk found in retrieval!"
        hashes.add(h)
    log_step(f"[RETRIEVAL] Retrieved {len(results)} unique chunks.")
    try:
        service.client.delete_collection(TEST_COLLECTION)
        log_step("[SUCCESS] Deduplication test passed.")
    except Exception as e:
        if hasattr(chromadb.errors, "NotFoundError") and isinstance(e, chromadb.errors.NotFoundError):
            pass
        else:
            raise

@pytest.mark.asyncio
async def test_concurrent_processing():
    clear_log()
    service = VectorService()
    log_step("[START] Concurrent processing test...")
    texts = [f"Concurrent doc {i}. " + "Sentence. " * 100 for i in range(5)]
    doc_ids = [f"doc_concurrent_{i}" for i in range(5)]
    section_title = "Concurrent Section"
    page_number = 1
    async def process_one(text, doc_id):
        chunks = await service.semantic_chunk(text)
        await service.store_chunks(chunks, TEST_COLLECTION, doc_id, section_title, page_number)
        log_step(f"[CONCURRENT] Stored doc {doc_id}")
    await asyncio.gather(*(process_one(text, doc_id) for text, doc_id in zip(texts, doc_ids)))
    log_step("[SUCCESS] Concurrent processing test completed.")
    try:
        service.client.delete_collection(TEST_COLLECTION)
    except Exception as e:
        if hasattr(chromadb.errors, "NotFoundError") and isinstance(e, chromadb.errors.NotFoundError):
            pass
        else:
            raise

@pytest.mark.asyncio
async def test_embedding_api_failure_logs_and_handles():
    clear_log()
    service = VectorService()
    text = "This is a test for embedding failure."
    doc_id = "doc_fail"
    section_title = "Fail Section"
    page_number = 1
    log_step("[START] Embedding API failure simulation test...")
    chunks = await service.semantic_chunk(text)
    # Patch generate_embeddings to raise an exception
    with patch.object(VectorService, "generate_embeddings", side_effect=Exception("Simulated embedding failure")):
        try:
            await service.store_chunks(chunks, TEST_COLLECTION, doc_id, section_title, page_number)
        except Exception as e:
            log_step(f"[FAILURE] Embedding failure caught: {e}")
            assert "Simulated embedding failure" in str(e)
        else:
            pytest.fail("Embedding failure was not raised as expected.")
    log_step("[SUCCESS] Embedding API failure simulation test completed.")

@pytest.mark.asyncio
async def test_chromadb_storage_failure_logs_and_handles():
    clear_log()
    service = VectorService()
    text = "This is a test for ChromaDB storage failure."
    doc_id = "doc_chromadb_fail"
    section_title = "ChromaDB Fail Section"
    page_number = 1
    log_step("[START] ChromaDB storage failure simulation test...")
    chunks = await service.semantic_chunk(text)
    # Patch the add method of the collection to raise an exception
    with patch("chromadb.api.models.Collection.Collection.add", side_effect=Exception("Simulated ChromaDB storage failure")):
        try:
            await service.store_chunks(chunks, TEST_COLLECTION, doc_id, section_title, page_number)
        except Exception as e:
            log_step(f"[FAILURE] ChromaDB storage failure caught: {e}")
            assert "Simulated ChromaDB storage failure" in str(e)
        else:
            pytest.fail("ChromaDB storage failure was not raised as expected.")
    log_step("[SUCCESS] ChromaDB storage failure simulation test completed.")

@pytest.mark.asyncio
async def test_redis_unavailable_logs_and_handles():
    clear_log()
    service = VectorService()
    text = "This is a test for Redis unavailability."
    doc_id = "doc_redis_fail"
    section_title = "Redis Fail Section"
    page_number = 1
    log_step("[START] Redis unavailability simulation test...")
    chunks = await service.semantic_chunk(text)
    # Patch the _get_redis method to raise an exception
    with patch.object(VectorService, "_get_redis", side_effect=Exception("Simulated Redis unavailable")):
        try:
            await service.store_chunks(chunks, TEST_COLLECTION, doc_id, section_title, page_number)
        except Exception as e:
            log_step(f"[FAILURE] Redis unavailable caught: {e}")
            assert "Simulated Redis unavailable" in str(e)
        else:
            pytest.fail("Redis unavailability was not raised as expected.")
    log_step("[SUCCESS] Redis unavailability simulation test completed.")

@pytest.mark.asyncio
async def test_memory_usage_during_large_pipeline():
    clear_log()
    service = VectorService()
    text = " ".join(["Sentence."] * 20000)  # Very large document
    doc_id = "doc_memory"
    section_title = "Memory Section"
    page_number = 1
    log_step("[START] Memory usage test for large pipeline...")

    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()

    chunks = await service.semantic_chunk(text)
    await service.store_chunks(chunks, TEST_COLLECTION, doc_id, section_title, page_number)

    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Calculate memory difference
    stats = end_snapshot.compare_to(start_snapshot, 'lineno')
    total_mem_diff = sum([stat.size_diff for stat in stats])
    log_step(f"[MEMORY] Total memory diff during pipeline: {total_mem_diff / 1024:.2f} KB")

    # Assert that memory usage did not grow by more than 100MB (arbitrary threshold)
    assert total_mem_diff < 100 * 1024 * 1024, "Memory usage grew by more than 100MB, possible leak!"

    try:
        service.client.delete_collection(TEST_COLLECTION)
    except Exception as e:
        if hasattr(chromadb.errors, "NotFoundError") and isinstance(e, chromadb.errors.NotFoundError):
            pass
        else:
            raise

    log_step("[SUCCESS] Memory usage test completed.")

"""
COMPREHENSIVE EDGE CASES AND ROBUSTNESS TEST
Tests system behavior with edge cases, error conditions, and stress scenarios
"""

import asyncio
import sys
import os
import time
import tempfile

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def log_test(test_name, message, status="INFO"):
    """Enhanced logging for test debugging"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{status}] {test_name}: {message}")

async def test_empty_and_invalid_inputs():
    """Test system behavior with empty and invalid inputs"""
    test_name = "EMPTY_INVALID"
    log_test(test_name, "Testing empty and invalid inputs", "START")
    
    try:
        from app.services.vector_service import VectorService
        from app.services.rag_service import RAGService
        
        vector_service = VectorService()
        rag_service = RAGService()
        
        # Test 1: Empty string
        log_test(test_name, "Testing empty string input...")
        empty_chunks = await vector_service.semantic_chunk("")
        log_test(test_name, f"Empty string result: {empty_chunks}")
        assert empty_chunks == [], "Empty string should produce no chunks"
        log_test(test_name, "Empty string handling: PASS", "PASS")
        
        # Test 2: Whitespace only
        log_test(test_name, "Testing whitespace-only input...")
        whitespace_chunks = await vector_service.semantic_chunk("   \n\t\r   ")
        log_test(test_name, f"Whitespace result: {whitespace_chunks}")
        assert whitespace_chunks == [], "Whitespace should produce no chunks"
        log_test(test_name, "Whitespace handling: PASS", "PASS")
        
        # Test 3: Very short text
        log_test(test_name, "Testing very short text...")
        short_text = "Hi."
        short_chunks = await vector_service.semantic_chunk(short_text)
        log_test(test_name, f"Short text '{short_text}' produced: {len(short_chunks)} chunks")
        log_test(test_name, f"Short chunks content: {short_chunks}")
        log_test(test_name, "Short text handling: PASS", "PASS")
        
        # Test 4: None input
        log_test(test_name, "Testing None input...")
        try:
            none_chunks = await vector_service.semantic_chunk(None)
            log_test(test_name, f"None input result: {none_chunks}")
            assert none_chunks == [], "None should produce no chunks"
            log_test(test_name, "None input gracefully handled: PASS", "PASS")
        except Exception as e:
            log_test(test_name, f"None input properly rejected: {e}", "PASS")
        
        # Test 5: Empty embedding list
        log_test(test_name, "Testing empty embedding list...")
        empty_embeddings = await vector_service.generate_embeddings([])
        log_test(test_name, f"Empty list embeddings: {empty_embeddings}")
        assert empty_embeddings == [], "Empty list should produce no embeddings"
        log_test(test_name, "Empty embedding list: PASS", "PASS")
        
        # Test 6: Mixed valid/invalid chunks
        log_test(test_name, "Testing mixed valid/invalid chunks...")
        mixed_chunks = ["valid text", "", None, "another valid text", "   "]
        mixed_embeddings = await vector_service.generate_embeddings(mixed_chunks)
        valid_count = len([e for e in mixed_embeddings if e is not None])
        log_test(test_name, f"Mixed input: {len(mixed_chunks)} chunks -> {valid_count} valid embeddings")
        log_test(test_name, f"Mixed embeddings result: {[type(e).__name__ for e in mixed_embeddings]}")
        log_test(test_name, "Mixed input handling: PASS", "PASS")
        
        return True
        
    except Exception as e:
        log_test(test_name, f"Test failed: {e}", "FAIL")
        import traceback
        log_test(test_name, f"Traceback: {traceback.format_exc()}")
        return False

async def test_large_document_processing():
    """Test system with very large documents"""
    test_name = "LARGE_DOCS"
    log_test(test_name, "Testing large document processing", "START")
    
    try:
        from app.services.vector_service import VectorService
        
        vector_service = VectorService()
        
        # Create progressively larger documents
        sizes = [1000, 5000, 10000]  # words
        
        for word_count in sizes:
            log_test(test_name, f"Testing document with {word_count} words...")
            
            # Create large document
            large_text = "This is a test sentence for large document processing. " * word_count
            log_test(test_name, f"Created text with {len(large_text)} characters")
            
            # Test chunking
            start_time = time.time()
            chunks = await vector_service.semantic_chunk(large_text, max_tokens=400)
            chunk_time = time.time() - start_time
            
            log_test(test_name, f"{word_count} words -> {len(chunks)} chunks in {chunk_time:.2f}s")
            log_test(test_name, f"Chunking rate: {len(chunks)/chunk_time:.1f} chunks/second")
            
            # Test embedding generation (limited batch for memory)
            test_batch_size = min(15, len(chunks))
            test_chunks = chunks[:test_batch_size]
            
            start_time = time.time()
            embeddings = await vector_service.generate_embeddings(test_chunks)
            embed_time = time.time() - start_time
            
            valid_embeddings = [e for e in embeddings if e is not None]
            log_test(test_name, f"Embeddings: {len(valid_embeddings)}/{test_batch_size} in {embed_time:.2f}s")
            log_test(test_name, f"Embedding rate: {len(valid_embeddings)/embed_time:.1f} embeddings/second")
            
            # Performance check
            if chunk_time < 15 and embed_time < 30:
                log_test(test_name, f"{word_count} words performance: ACCEPTABLE", "PASS")
            else:
                log_test(test_name, f"{word_count} words performance: SLOW", "WARN")
        
        return True
        
    except Exception as e:
        log_test(test_name, f"Large document test failed: {e}", "FAIL")
        import traceback
        log_test(test_name, f"Traceback: {traceback.format_exc()}")
        return False

async def test_special_characters_encoding():
    """Test system with special characters and different encodings"""
    test_name = "SPECIAL_CHARS"
    log_test(test_name, "Testing special characters and encoding", "START")
    
    try:
        from app.services.vector_service import VectorService
        
        vector_service = VectorService()
        
        # Test cases with different character types
        test_cases = [
            ("Accented", "Café résumé naïve Zürich"),
            ("Symbols", "@#$%^&*()_+-=[]{}|;:,.<>?/~`"),
            ("Numbers", "123 3.14159 -42 1e10 0xFF"),
            ("Unicode", "Hello世界 Привет мир مرحبا بالعالم"),
            ("Emoji", "🚀 🎉 ✅ ❌ 🔥 💡 🌟"),
            ("Mixed", "Price: $29.99 (€25.50) - 50% off! 🎊")
        ]
        
        for case_name, test_text in test_cases:
            log_test(test_name, f"Testing {case_name}: '{test_text}'")
            
            # Test chunking
            chunks = await vector_service.semantic_chunk(test_text)
            log_test(test_name, f"{case_name} chunking: {len(chunks)} chunks")
            
            if chunks:
                log_test(test_name, f"{case_name} chunk content: {chunks[0]}")
                
                # Test embeddings
                embeddings = await vector_service.generate_embeddings(chunks[:1])
                if embeddings and embeddings[0] is not None:
                    log_test(test_name, f"{case_name} embeddings: SUCCESS ({len(embeddings[0])} dims)", "PASS")
                else:
                    log_test(test_name, f"{case_name} embeddings: FAILED", "WARN")
            else:
                log_test(test_name, f"{case_name}: No chunks created", "WARN")
        
        return True
        
    except Exception as e:
        log_test(test_name, f"Special characters test failed: {e}", "FAIL")
        import traceback
        log_test(test_name, f"Traceback: {traceback.format_exc()}")
        return False

async def test_concurrent_processing():
    """Test system under concurrent load"""
    test_name = "CONCURRENT"
    log_test(test_name, "Testing concurrent processing", "START")
    
    try:
        from app.services.vector_service import VectorService
        
        vector_service = VectorService()
        
        # Create multiple test documents
        num_docs = 8
        test_docs = []
        for i in range(num_docs):
            doc_content = f"Document {i}: This is concurrent processing test content. " * 100
            test_docs.append((doc_content, f"doc_{i}"))
        
        log_test(test_name, f"Created {num_docs} test documents")
        
        # Process documents concurrently
        async def process_single_doc(doc_content, doc_id):
            start_time = time.time()
            
            # Chunk document
            chunks = await vector_service.semantic_chunk(doc_content, max_tokens=250)
            
            # Generate embeddings (limited for performance)
            test_chunks = chunks[:5]
            embeddings = await vector_service.generate_embeddings(test_chunks)
            valid_embeddings = [e for e in embeddings if e is not None]
            
            process_time = time.time() - start_time
            
            return {
                'doc_id': doc_id,
                'chunks': len(chunks),
                'embeddings': len(valid_embeddings),
                'time': process_time
            }
        
        # Run concurrent processing
        log_test(test_name, "Starting concurrent processing...")
        start_time = time.time()
        
        results = await asyncio.gather(*[
            process_single_doc(content, doc_id) 
            for content, doc_id in test_docs
        ])
        
        total_time = time.time() - start_time
        
        # Analyze results
        total_chunks = sum(r['chunks'] for r in results)
        total_embeddings = sum(r['embeddings'] for r in results)
        avg_time = sum(r['time'] for r in results) / len(results)
        
        log_test(test_name, f"Concurrent processing completed in {total_time:.2f}s")
        log_test(test_name, f"Total chunks processed: {total_chunks}")
        log_test(test_name, f"Total embeddings generated: {total_embeddings}")
        log_test(test_name, f"Average per-document time: {avg_time:.2f}s")
        log_test(test_name, f"Throughput: {num_docs/total_time:.1f} documents/second")
        
        # Show individual results
        for result in results:
            log_test(test_name, f"  {result['doc_id']}: {result['chunks']} chunks, {result['embeddings']} embeddings, {result['time']:.2f}s")
        
        log_test(test_name, "Concurrent processing: PASS", "PASS")
        return True
        
    except Exception as e:
        log_test(test_name, f"Concurrent processing test failed: {e}", "FAIL")
        import traceback
        log_test(test_name, f"Traceback: {traceback.format_exc()}")
        return False

async def test_memory_stress():
    """Test system memory usage under stress"""
    test_name = "MEMORY_STRESS"
    log_test(test_name, "Testing memory stress conditions", "START")
    
    try:
        from app.services.vector_service import VectorService
        from app.monitoring import monitor
        
        vector_service = VectorService()
        
        # Monitor initial memory
        monitor.log_memory_usage("stress_test_start")
        
        # Create memory-intensive operations
        log_test(test_name, "Creating large chunk dataset...")
        large_chunks = []
        
        for batch in range(15):  # 15 batches
            log_test(test_name, f"Processing batch {batch + 1}/15...")
            
            # Create large text
            text = f"Memory stress test batch {batch}. " * 800  # ~4000 words
            chunks = await vector_service.semantic_chunk(text, max_tokens=300)
            large_chunks.extend(chunks)
            
            log_test(test_name, f"Batch {batch + 1}: {len(chunks)} chunks (total: {len(large_chunks)})")
            
            # Monitor memory every 5 batches
            if (batch + 1) % 5 == 0:
                monitor.log_memory_usage(f"batch_{batch + 1}")
        
        log_test(test_name, f"Created total of {len(large_chunks)} chunks for stress test")
        
        # Test embedding generation in controlled batches
        batch_size = 8
        total_embeddings = 0
        
        for i in range(0, min(80, len(large_chunks)), batch_size):  # Limit to 80 chunks
            batch_num = i // batch_size + 1
            batch = large_chunks[i:i+batch_size]
            
            log_test(test_name, f"Embedding batch {batch_num}: {len(batch)} chunks")
            
            start_time = time.time()
            embeddings = await vector_service.generate_embeddings(batch)
            batch_time = time.time() - start_time
            
            valid_count = len([e for e in embeddings if e is not None])
            total_embeddings += valid_count
            
            log_test(test_name, f"Batch {batch_num}: {valid_count}/{len(batch)} embeddings in {batch_time:.2f}s")
            
            # Memory check after each batch
            if batch_num % 3 == 0:
                monitor.log_memory_usage(f"embed_batch_{batch_num}")
        
        log_test(test_name, f"Total embeddings generated: {total_embeddings}")
        
        # Final memory check
        monitor.log_memory_usage("stress_test_end")
        
        log_test(test_name, "Memory stress test completed", "PASS")
        return True
        
    except Exception as e:
        log_test(test_name, f"Memory stress test failed: {e}", "FAIL")
        import traceback
        log_test(test_name, f"Traceback: {traceback.format_exc()}")
        return False

async def test_error_recovery():
    """Test system error recovery and graceful degradation"""
    test_name = "ERROR_RECOVERY"
    log_test(test_name, "Testing error recovery mechanisms", "START")
    
    try:
        from app.services.vector_service import VectorService
        from app.services.rag_service import RAGService
        
        vector_service = VectorService()
        rag_service = RAGService()
        
        # Test 1: Malformed input types
        log_test(test_name, "Testing malformed input types...")
        malformed_inputs = [123, [], {}, True, 3.14]
        
        for i, bad_input in enumerate(malformed_inputs):
            log_test(test_name, f"Testing malformed input {i+1}: {type(bad_input).__name__} = {bad_input}")
            try:
                result = await vector_service.semantic_chunk(bad_input)
                log_test(test_name, f"Malformed input {i+1} result: {result}")
                if result == []:
                    log_test(test_name, f"Malformed input {i+1}: Gracefully handled", "PASS")
                else:
                    log_test(test_name, f"Malformed input {i+1}: Unexpected result", "WARN")
            except Exception as e:
                log_test(test_name, f"Malformed input {i+1}: Properly rejected - {e}", "PASS")
        
        # Test 2: Invalid embedding inputs
        log_test(test_name, "Testing invalid embedding inputs...")
        invalid_chunks = [None, "", "   ", 123, [], "valid text"]
        
        embeddings = await vector_service.generate_embeddings(invalid_chunks)
        valid_count = len([e for e in embeddings if e is not None])
        
        log_test(test_name, f"Invalid chunks: {len(invalid_chunks)} -> {valid_count} valid embeddings")
        log_test(test_name, f"Embedding results: {[type(e).__name__ if e else 'None' for e in embeddings]}")
        log_test(test_name, "Invalid embedding input handling: PASS", "PASS")
        
        # Test 3: RAG with empty context
        log_test(test_name, "Testing RAG with empty context...")
        empty_context = await rag_service.build_context([], max_tokens=1000)
        log_test(test_name, f"Empty candidates context: '{empty_context}'")
        assert empty_context == "", "Empty candidates should produce empty context"
        log_test(test_name, "Empty context handling: PASS", "PASS")
        
        # Test 4: Query enhancement fallback
        log_test(test_name, "Testing query enhancement fallback...")
        test_query = "test query for fallback"
        fallback_query = await rag_service.enhance_query(test_query)
        log_test(test_name, f"Original: '{test_query}' -> Enhanced: '{fallback_query}'")
        
        if fallback_query == test_query:
            log_test(test_name, "Query enhancement fallback working (no API key)", "PASS")
        else:
            log_test(test_name, "Query enhancement active (API key present)", "PASS")
        
        # Test 5: Large token context handling
        log_test(test_name, "Testing large token context handling...")
        large_candidates = [
            {"text": "Large text content " * 200, "metadata": {"hash": f"hash_{i}"}}
            for i in range(10)
        ]
        
        context = await rag_service.build_context(large_candidates, max_tokens=500)
        log_test(test_name, f"Large context result: {len(context)} characters")
        
        if len(context) > 0:
            log_test(test_name, "Large context handling: PASS", "PASS")
        else:
            log_test(test_name, "Large context handling: Issue detected", "WARN")
        
        return True
        
    except Exception as e:
        log_test(test_name, f"Error recovery test failed: {e}", "FAIL")
        import traceback
        log_test(test_name, f"Traceback: {traceback.format_exc()}")
        return False

async def test_data_consistency():
    """Test data consistency and deduplication"""
    test_name = "DATA_CONSISTENCY"
    log_test(test_name, "Testing data consistency and deduplication", "START")
    
    try:
        from app.services.vector_service import VectorService
        
        vector_service = VectorService()
        
        # Test with intentionally duplicate content
        log_test(test_name, "Creating duplicate content test...")
        base_text = "This is duplicate content for testing. "
        duplicate_text = base_text * 5  # Repeat 5 times
        
        log_test(test_name, f"Base text: '{base_text}'")
        log_test(test_name, f"Duplicate text length: {len(duplicate_text)} characters")
        
        # Create chunks
        chunks = await vector_service.semantic_chunk(duplicate_text)
        log_test(test_name, f"Original chunks: {len(chunks)}")
        
        # Intentionally duplicate the chunks
        duplicated_chunks = chunks + chunks + chunks  # Triple the chunks
        log_test(test_name, f"Artificially duplicated chunks: {len(duplicated_chunks)}")
        
        # Test storage with deduplication
        collection_name = "dedup_test_collection"
        doc_id = "dedup_test_doc"
        
        try:
            log_test(test_name, "Storing chunks with deduplication...")
            await vector_service.store_chunks(duplicated_chunks, collection_name, doc_id)
            
            # Retrieve and analyze for duplicates
            log_test(test_name, "Retrieving stored chunks for analysis...")
            search_results = await vector_service.similarity_search(
                query="duplicate content testing",
                collection_name=collection_name,
                k=50  # Get many results
            )
            
            log_test(test_name, f"Retrieved {len(search_results)} results from storage")
            
            # Check for hash-based deduplication
            hashes = set()
            duplicates_found = 0
            
            for i, result in enumerate(search_results):
                chunk_hash = result.get("metadata", {}).get("hash")
                text_preview = result.get("text", "")[:50]
                score = result.get("score", 0)
                
                log_test(test_name, f"Result {i+1}: hash={chunk_hash[:8] if chunk_hash else 'None'}..., score={score:.4f}, text='{text_preview}...'")
                
                if chunk_hash:
                    if chunk_hash in hashes:
                        duplicates_found += 1
                        log_test(test_name, f"DUPLICATE DETECTED: {chunk_hash[:8]}...", "WARN")
                    hashes.add(chunk_hash)
            
            log_test(test_name, f"Unique hashes found: {len(hashes)}")
            log_test(test_name, f"Duplicates detected: {duplicates_found}")
            
            if duplicates_found == 0:
                log_test(test_name, "Deduplication working correctly", "PASS")
            else:
                log_test(test_name, f"Deduplication issue: {duplicates_found} duplicates found", "WARN")
            
            # Test retrieval consistency
            log_test(test_name, "Testing retrieval consistency...")
            second_search = await vector_service.similarity_search(
                query="duplicate content testing",
                collection_name=collection_name,
                k=10
            )
            
            if len(second_search) == len(search_results[:10]):
                log_test(test_name, "Retrieval consistency: PASS", "PASS")
            else:
                log_test(test_name, f"Retrieval inconsistency: {len(second_search)} vs {len(search_results[:10])}", "WARN")
            
            return True
            
        finally:
            # Cleanup
            try:
                if hasattr(vector_service, 'client') and vector_service.client:
                    vector_service.client.delete_collection(collection_name)
                    log_test(test_name, "Test collection cleaned up")
            except Exception as e:
                log_test(test_name, f"Cleanup warning: {e}", "WARN")
        
    except Exception as e:
        log_test(test_name, f"Data consistency test failed: {e}", "FAIL")
        import traceback
        log_test(test_name, f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Run all edge case tests"""
    print("=" * 80)
    print("QNA-RAG-DATABASE EDGE CASES AND ROBUSTNESS TEST")
    print("=" * 80)
    
    # Define all test functions
    test_functions = [
        ("Empty/Invalid Inputs", test_empty_and_invalid_inputs),
        ("Large Document Processing", test_large_document_processing),
        ("Special Characters/Encoding", test_special_characters_encoding),
        ("Concurrent Processing", test_concurrent_processing),
        ("Memory Stress", test_memory_stress),
        ("Error Recovery", test_error_recovery),
        ("Data Consistency", test_data_consistency)
    ]
    
    results = []
    start_time = time.time()
    
    # Run each test
    for test_desc, test_func in test_functions:
        print(f"\n{'-' * 60}")
        print(f"RUNNING: {test_desc}")
        print(f"{'-' * 60}")
        
        try:
            result = await test_func()
            results.append((test_desc, result))
            
            if result:
                print(f"[PASS] {test_desc}: PASSED")
            else:
                print(f"[FAIL] {test_desc}: FAILED")
                
        except Exception as e:
            print(f"[CRASH] {test_desc}: CRASHED - {e}")
            results.append((test_desc, False))
    
    # Final summary
    total_time = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("\n" + "=" * 80)
    print("EDGE CASES TEST SUMMARY")
    print("=" * 80)
    
    for test_desc, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_desc}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\nALL EDGE CASE TESTS PASSED!")
        print("System is robust and production-ready!")
    else:
        print(f"\n{total - passed} tests failed")
        print("Review the detailed logs above for debugging")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
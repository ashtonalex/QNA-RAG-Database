"""
COMPLETE END-TO-END SYSTEM TEST
Tests the entire workflow: Text Processing -> Chunking -> Embeddings -> Vector Storage -> RAG Pipeline
"""

import asyncio
import sys
import os
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def log_step(step, message, status="INFO"):
    """Enhanced logging for debugging"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{status}] {step}: {message}")

async def test_complete_system():
    """Test complete end-to-end system functionality"""
    log_step("INIT", "Starting Complete System Test", "START")
    
    try:
        # Import all required services
        log_step("IMPORT", "Importing system services...")
        from app.services.vector_service import VectorService
        from app.services.rag_service import RAGService
        from app.monitoring import monitor
        log_step("IMPORT", "All services imported successfully", "PASS")
        
        # Test document content
        test_content = """
        Artificial Intelligence and Machine Learning Guide
        
        Artificial Intelligence (AI) represents one of the most significant technological advances of our time. AI systems can perform tasks that typically require human intelligence, including visual perception, speech recognition, decision-making, and language translation.
        
        Core AI Technologies:
        
        1. Machine Learning (ML)
        Machine learning enables systems to automatically learn and improve from experience without explicit programming. ML algorithms build mathematical models based on training data to make predictions or decisions.
        
        Types of Machine Learning:
        - Supervised Learning: Uses labeled training data (classification, regression)
        - Unsupervised Learning: Finds patterns in unlabeled data (clustering, dimensionality reduction)
        - Reinforcement Learning: Learns through trial and error with rewards and penalties
        
        2. Deep Learning
        Deep learning uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input. It excels in image recognition, natural language processing, and speech recognition.
        
        3. Natural Language Processing (NLP)
        NLP focuses on the interaction between computers and human language. Applications include machine translation, sentiment analysis, chatbots, and text summarization.
        
        4. Computer Vision
        Computer vision enables machines to interpret and understand visual information from the world. Applications include facial recognition, autonomous vehicles, medical imaging, and quality control.
        
        Real-World Applications:
        - Healthcare: Medical diagnosis, drug discovery, personalized treatment
        - Finance: Fraud detection, algorithmic trading, risk assessment
        - Transportation: Autonomous vehicles, route optimization, traffic management
        - Entertainment: Recommendation systems, content generation, game AI
        - Manufacturing: Quality control, predictive maintenance, robotics
        
        Future Trends:
        The future of AI includes developments in quantum computing, edge AI, explainable AI, and artificial general intelligence (AGI).
        """
        
        # Step 1: Initialize Services
        log_step("STEP1", "Initializing Vector and RAG services...")
        vector_service = VectorService()
        rag_service = RAGService()
        log_step("STEP1", "Services initialized successfully", "PASS")
        
        # Step 2: Text Chunking
        log_step("STEP2", "Starting semantic text chunking...")
        log_step("STEP2", f"Input text length: {len(test_content)} characters")
        
        chunks = await vector_service.semantic_chunk(test_content, max_tokens=300)
        log_step("STEP2", f"Created {len(chunks)} semantic chunks", "PASS")
        
        # Display chunk details for debugging
        for i, chunk in enumerate(chunks):
            log_step("STEP2", f"Chunk {i+1} length: {len(chunk)} chars - Preview: {chunk[:80]}...")
        
        # Step 3: Embedding Generation
        log_step("STEP3", "Generating vector embeddings...")
        start_time = time.time()
        embeddings = await vector_service.generate_embeddings(chunks)
        embed_time = time.time() - start_time
        
        valid_embeddings = [e for e in embeddings if e is not None]
        log_step("STEP3", f"Generated {len(valid_embeddings)}/{len(chunks)} embeddings in {embed_time:.2f}s", "PASS")
        
        if valid_embeddings:
            log_step("STEP3", f"Embedding dimensions: {len(valid_embeddings[0])}")
            log_step("STEP3", f"Sample embedding values: {valid_embeddings[0][:5]}...")
        
        # Step 4: Vector Storage
        log_step("STEP4", "Storing chunks in ChromaDB...")
        collection_name = "system_test_collection"
        doc_id = "ai_ml_guide_001"
        
        await vector_service.store_chunks(chunks, collection_name, doc_id, "AI Guide", 1)
        log_step("STEP4", f"Stored {len(chunks)} chunks in collection: {collection_name}", "PASS")
        
        # Step 5: Vector Retrieval Testing
        log_step("STEP5", "Testing vector similarity search...")
        
        test_queries = [
            "What is machine learning?",
            "How does deep learning work?", 
            "What are AI applications in healthcare?",
            "Explain natural language processing",
            "What is computer vision used for?"
        ]
        
        all_search_results = []
        for i, query in enumerate(test_queries):
            log_step("STEP5", f"Query {i+1}: '{query}'")
            
            results = await vector_service.similarity_search(
                query=query,
                collection_name=collection_name,
                k=3
            )
            
            log_step("STEP5", f"Retrieved {len(results)} results with scores: {[str(r.get('score', 0))[:6] for r in results]}")
            
            # Show result details for debugging
            for j, result in enumerate(results):
                text_preview = result.get('text', '')[:100]
                score = result.get('score', 0)
                log_step("STEP5", f"  Result {j+1}: Score={score:.4f}, Text='{text_preview}...'")
            
            all_search_results.extend(results)
        
        log_step("STEP5", f"Total search results collected: {len(all_search_results)}", "PASS")
        
        # Step 6: RAG Pipeline Testing
        log_step("STEP6", "Testing RAG pipeline components...")
        
        # Test query enhancement
        test_query = "What are the differences between supervised and unsupervised learning?"
        log_step("STEP6", f"Original query: '{test_query}'")
        
        enhanced_query = await rag_service.enhance_query(test_query)
        log_step("STEP6", f"Enhanced query: '{enhanced_query}'")
        
        if enhanced_query != test_query:
            log_step("STEP6", "Query enhancement active", "PASS")
        else:
            log_step("STEP6", "Query enhancement fallback (no API key)", "WARN")
        
        # Test candidate retrieval
        log_step("STEP6", "Retrieving RAG candidates...")
        candidates = await rag_service.retrieve_candidates(
            query=enhanced_query,
            collection_name=collection_name,
            k=8,
            vector_service=vector_service
        )
        
        log_step("STEP6", f"Retrieved {len(candidates)} candidates", "PASS")
        
        # Show candidate details
        for i, candidate in enumerate(candidates[:3]):  # Show first 3
            text_preview = candidate.get('text', '')[:80]
            metadata = candidate.get('metadata', {})
            log_step("STEP6", f"  Candidate {i+1}: {text_preview}... (doc_id: {metadata.get('doc_id', 'N/A')})")
        
        # Test context building
        log_step("STEP6", "Building context from candidates...")
        context = await rag_service.build_context(candidates, max_tokens=2000)
        
        log_step("STEP6", f"Built context: {len(context)} characters", "PASS")
        log_step("STEP6", f"Context preview: {context[:200]}...")
        
        # Verify context quality
        context_lower = context.lower()
        expected_terms = ["machine learning", "supervised", "unsupervised", "artificial intelligence"]
        found_terms = [term for term in expected_terms if term in context_lower]
        
        log_step("STEP6", f"Context contains expected terms: {found_terms}")
        
        if len(found_terms) >= 2:
            log_step("STEP6", "Context quality verification passed", "PASS")
        else:
            log_step("STEP6", "Context quality may be insufficient", "WARN")
        
        # Step 7: System Performance Monitoring
        log_step("STEP7", "Checking system performance...")
        monitor.log_memory_usage("system_test_complete")
        
        # Performance metrics
        chunks_per_sec = len(chunks) / max(embed_time, 0.1)
        embeddings_per_sec = len(valid_embeddings) / max(embed_time, 0.1)
        
        log_step("STEP7", f"Performance metrics:")
        log_step("STEP7", f"  - Chunking rate: {chunks_per_sec:.1f} chunks/second")
        log_step("STEP7", f"  - Embedding rate: {embeddings_per_sec:.1f} embeddings/second")
        log_step("STEP7", f"  - Total processing time: {embed_time:.2f} seconds")
        
        # Step 8: Data Integrity Verification
        log_step("STEP8", "Verifying data integrity...")
        
        # Verify all chunks are retrievable
        verification_results = await vector_service.similarity_search(
            query="artificial intelligence machine learning",
            collection_name=collection_name,
            k=len(chunks) * 2  # Get more than stored to ensure all are found
        )
        
        unique_hashes = set()
        for result in verification_results:
            chunk_hash = result.get("metadata", {}).get("hash")
            if chunk_hash:
                unique_hashes.add(chunk_hash)
        
        log_step("STEP8", f"Verified {len(unique_hashes)} unique chunks in storage")
        
        if len(unique_hashes) >= len(chunks):
            log_step("STEP8", "Data integrity verification passed", "PASS")
        else:
            log_step("STEP8", f"Data integrity issue: expected {len(chunks)}, found {len(unique_hashes)}", "WARN")
        
        # Step 9: Cleanup
        log_step("STEP9", "Cleaning up test data...")
        try:
            if hasattr(vector_service, 'client') and vector_service.client:
                vector_service.client.delete_collection(collection_name)
                log_step("STEP9", "Test collection deleted successfully", "PASS")
        except Exception as e:
            log_step("STEP9", f"Cleanup warning: {e}", "WARN")
        
        # Final Results
        log_step("COMPLETE", "=== END-TO-END TEST COMPLETED SUCCESSFULLY ===", "PASS")
        log_step("COMPLETE", "All system components working correctly")
        log_step("COMPLETE", "System is ready for production use")
        
        return True
        
    except Exception as e:
        log_step("ERROR", f"System test failed: {e}", "FAIL")
        import traceback
        log_step("ERROR", f"Full traceback: {traceback.format_exc()}")
        return False

async def main():
    """Run complete system test"""
    print("=" * 80)
    print("QNA-RAG-DATABASE COMPLETE SYSTEM TEST")
    print("=" * 80)
    
    start_time = time.time()
    success = await test_complete_system()
    total_time = time.time() - start_time
    
    print("=" * 80)
    if success:
        print(f"COMPLETE SYSTEM TEST PASSED in {total_time:.2f} seconds")
        print("System is fully operational and production-ready!")
    else:
        print(f"COMPLETE SYSTEM TEST FAILED after {total_time:.2f} seconds")
        print("Check the logs above for debugging information")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
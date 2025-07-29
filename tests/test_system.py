"""
Comprehensive system test for the complete RAG pipeline.
Tests end-to-end functionality from document upload to response generation.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from backend.app.services.enhanced_rag_pipeline import EnhancedRAGPipeline, PipelineConfig
from backend.app.models.chunk_models import Chunk


class TestRAGSystem:
    """Complete system test for RAG pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Enhanced RAG pipeline fixture"""
        config = PipelineConfig(
            max_candidates=10,
            rerank_top_n=5,
            timeout_seconds=30
        )
        return EnhancedRAGPipeline(config)
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {
        'OPENROUTER_API_KEY': 'test-openrouter-key',
        'cohere_API_key': 'test-cohere-key'
    })
    async def test_complete_rag_system(self, pipeline):
        """Test complete RAG system from query to response"""
        
        # Mock data
        test_query = "What is Python programming?"
        enhanced_query = "Python programming language features syntax"
        
        mock_candidates = [
            {
                "text": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {"doc_id": "python_guide.pdf", "page_number": 1, "hash": "hash1"},
                "score": 0.95
            },
            {
                "text": "Python supports object-oriented programming and has extensive libraries.",
                "metadata": {"doc_id": "python_guide.pdf", "page_number": 2, "hash": "hash2"},
                "score": 0.88
            }
        ]
        
        expected_response = "Python is a versatile, high-level programming language that emphasizes code readability and simplicity. It supports multiple programming paradigms and is widely used for various applications."
        
        # Mock all pipeline components
        with patch.object(pipeline.rag_service, 'enhance_query', return_value=enhanced_query), \
             patch.object(pipeline.rag_service, 'retrieve_candidates', return_value=mock_candidates), \
             patch.object(pipeline.cohere_reranker, 'score', return_value=[0.92, 0.85]), \
             patch.object(pipeline.cross_reranker, 'score', return_value=[0.89, 0.83]), \
             patch.object(pipeline.rag_service, 'generate_response', return_value=expected_response):
            
            # Execute complete pipeline
            result = await pipeline.process_query(
                query=test_query,
                collection_name="test_collection"
            )
            
            # Verify results
            assert result["response"] == expected_response
            assert len(result["sources"]) > 0
            assert result["pipeline_info"]["original_query"] == test_query
            assert result["pipeline_info"]["enhanced_query"] == enhanced_query
            assert result["pipeline_info"]["candidates_found"] == 2
            assert result["pipeline_info"]["reranked_chunks"] > 0
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {
        'OPENROUTER_API_KEY': 'test-openrouter-key',
        'cohere_API_key': 'test-cohere-key'
    })
    async def test_system_error_handling(self, pipeline):
        """Test system error handling and fallbacks"""
        
        # Test with no candidates found
        with patch.object(pipeline.rag_service, 'enhance_query', return_value="enhanced query"), \
             patch.object(pipeline.rag_service, 'retrieve_candidates', return_value=[]):
            
            result = await pipeline.process_query(
                query="No results query",
                collection_name="empty_collection"
            )
            
            assert "couldn't find relevant information" in result["response"]
            assert result["sources"] == []
            assert result["pipeline_info"]["candidates_found"] == 0
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, pipeline):
        """Test system health monitoring"""
        
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key',
            'cohere_API_key': 'test-key'
        }):
            health = await pipeline.health_check()
            
            assert health["overall_status"] == "healthy"
            assert "components" in health
            assert "configuration" in health
            assert health["components"]["openrouter_api"]["status"] == "ok"
            assert health["components"]["cohere_api"]["status"] == "ok"
    
    def test_pipeline_configuration(self):
        """Test pipeline configuration options"""
        
        custom_config = PipelineConfig(
            max_candidates=20,
            rerank_top_n=10,
            cohere_weight=0.7,
            cross_weight=0.3
        )
        
        pipeline = EnhancedRAGPipeline(custom_config)
        
        assert pipeline.config.max_candidates == 20
        assert pipeline.config.rerank_top_n == 10
        assert pipeline.config.cohere_weight == 0.7
        assert pipeline.config.cross_weight == 0.3
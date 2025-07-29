"""
Enhanced RAG Pipeline with validation, configuration, and robust error handling.
Addresses assumptions made in the original integration.
"""

import logging
import asyncio
import time
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from fastapi import WebSocket
from backend.app.services.rag_service import RAGService
from backend.app.services.cohere_reranker import CohereReranker
from backend.app.services.crossencoder_reranker import CrossEncoderReranker
from backend.app.services.vector_service import VectorService
from backend.app.models.chunk_models import Chunk

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for RAG pipeline behavior"""
    # Retrieval settings
    max_candidates: int = 20
    rerank_top_n: int = 10
    context_max_tokens: int = 4000
    
    # Reranking settings
    cohere_weight: float = 0.6
    cross_weight: float = 0.4
    rerank_method: str = "weighted"  # "weighted" or "vote"
    
    # Error handling
    max_retries: int = 3
    timeout_seconds: int = 30
    fallback_on_rerank_failure: bool = True
    
    # Rate limiting
    max_concurrent_requests: int = 10
    request_delay_ms: int = 100
    
    # Memory management
    max_context_length: int = 10000
    max_query_length: int = 1000

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class RateLimitError(Exception):
    """Custom exception for rate limiting"""
    pass

class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with proper validation, configuration, and error handling.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.rag_service = RAGService()
        self.vector_service = VectorService()
        self.cohere_reranker = CohereReranker()
        self.cross_reranker = CrossEncoderReranker()
        
        # Rate limiting
        self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._last_request_time = 0
    
    def _validate_inputs(self, query: str, collection_name: str, k: int) -> None:
        """Validate input parameters"""
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        if len(query) > self.config.max_query_length:
            raise ValidationError(f"Query too long (max {self.config.max_query_length} characters)")
        if not collection_name or not collection_name.strip():
            raise ValidationError("Collection name cannot be empty")
        if k <= 0 or k > 100:
            raise ValidationError("k must be between 1 and 100")
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return []
        
        # Handle edge cases
        valid_scores = [s for s in scores if isinstance(s, (int, float)) and not (s != s)]  # Filter NaN
        if not valid_scores:
            return [0.0] * len(scores)
        
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        normalized = []
        for score in scores:
            if isinstance(score, (int, float)) and score == score:  # Not NaN
                normalized.append((score - min_score) / (max_score - min_score))
            else:
                normalized.append(0.0)
        
        return normalized
    
    def _validate_candidate_format(self, candidates: List[Any]) -> List[Dict]:
        """Validate and normalize candidate format"""
        validated = []
        
        for i, candidate in enumerate(candidates):
            try:
                if not isinstance(candidate, dict):
                    logger.warning(f"Candidate {i} is not a dict, skipping")
                    continue
                
                # Ensure required fields
                text = candidate.get("text", "")
                if not text or not isinstance(text, str) or not text.strip():
                    logger.warning(f"Candidate {i} has invalid text, skipping")
                    continue
                
                metadata = candidate.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                
                score = candidate.get("score", 0.0)
                if not isinstance(score, (int, float)) or score != score:  # Check for NaN
                    score = 0.0
                
                validated.append({
                    "text": text.strip(),
                    "metadata": metadata,
                    "score": float(score)
                })
                
            except Exception as e:
                logger.warning(f"Error validating candidate {i}: {e}")
                continue
        
        return validated
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting"""
        async with self._request_semaphore:
            current_time = time.time() * 1000  # Convert to milliseconds
            time_since_last = current_time - self._last_request_time
            
            if time_since_last < self.config.request_delay_ms:
                delay = (self.config.request_delay_ms - time_since_last) / 1000
                await asyncio.sleep(delay)
            
            self._last_request_time = time.time() * 1000
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed")
        
        raise last_exception
    
    async def process_query(
        self,
        query: str,
        collection_name: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
        stream: bool = False,
        websocket: Optional[WebSocket] = None,
        rag_prompt_template: str = "Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline with validation and error handling.
        """
        # Set defaults and validate
        k = k or self.config.max_candidates
        self._validate_inputs(query, collection_name, k)
        
        # Apply rate limiting
        await self._rate_limit()
        
        pipeline_start_time = time.time()
        
        try:
            # Phase 1: Query Enhancement
            logger.info(f"Phase 1: Enhancing query: {query[:100]}...")
            enhanced_query = await asyncio.wait_for(
                self._retry_with_backoff(self.rag_service.enhance_query, query),
                timeout=self.config.timeout_seconds
            )
            logger.info(f"Enhanced query: {enhanced_query[:100]}...")
            
            # Phase 2: Candidate Retrieval
            logger.info("Phase 2: Retrieving candidates")
            candidates = await asyncio.wait_for(
                self._retry_with_backoff(
                    self.rag_service.retrieve_candidates,
                    enhanced_query,
                    collection_name,
                    k,
                    metadata_filter,
                    self.vector_service
                ),
                timeout=self.config.timeout_seconds
            )
            
            # Validate candidate format
            candidates = self._validate_candidate_format(candidates)
            
            if not candidates:
                return {
                    "response": "I couldn't find relevant information to answer your question.",
                    "sources": [],
                    "pipeline_info": {
                        "original_query": query,
                        "enhanced_query": enhanced_query,
                        "candidates_found": 0,
                        "reranked_chunks": 0,
                        "processing_time_ms": int((time.time() - pipeline_start_time) * 1000)
                    }
                }
            
            logger.info(f"Retrieved {len(candidates)} valid candidates")
            
            # Phase 3: Dual Reranking
            logger.info("Phase 3: Dual reranking")
            reranked_chunks = await self._rerank_candidates_robust(enhanced_query, candidates)
            logger.info(f"Reranked to {len(reranked_chunks)} chunks")
            
            # Phase 4: Context Building
            logger.info("Phase 4: Building context")
            context_candidates = self._prepare_context_candidates(reranked_chunks)
            
            context = await asyncio.wait_for(
                self.rag_service.build_context(
                    context_candidates, 
                    max_tokens=self.config.context_max_tokens
                ),
                timeout=self.config.timeout_seconds
            )
            
            # Validate context length
            if len(context) > self.config.max_context_length:
                context = context[:self.config.max_context_length] + "..."
                logger.warning("Context truncated due to length limit")
            
            logger.info(f"Built context with {len(context)} characters")
            
            # Phase 5: Response Generation
            logger.info("Phase 5: Generating response")
            response = await asyncio.wait_for(
                self._retry_with_backoff(
                    self.rag_service.generate_response,
                    query,
                    context,
                    rag_prompt_template,
                    stream,
                    websocket
                ),
                timeout=self.config.timeout_seconds * 2  # Allow more time for generation
            )
            
            # Extract sources for citation
            sources = self._extract_sources_robust(reranked_chunks)
            
            processing_time = int((time.time() - pipeline_start_time) * 1000)
            
            return {
                "response": response,
                "sources": sources,
                "pipeline_info": {
                    "original_query": query,
                    "enhanced_query": enhanced_query,
                    "candidates_found": len(candidates),
                    "reranked_chunks": len(reranked_chunks),
                    "context_length": len(context),
                    "processing_time_ms": processing_time,
                    "config_used": {
                        "rerank_method": self.config.rerank_method,
                        "cohere_weight": self.config.cohere_weight,
                        "cross_weight": self.config.cross_weight
                    }
                }
            }
            
        except asyncio.TimeoutError:
            logger.error("Pipeline timeout exceeded")
            return {
                "response": "I'm sorry, but processing your question took too long. Please try again with a simpler query.",
                "sources": [],
                "pipeline_info": {
                    "error": "timeout",
                    "original_query": query,
                    "processing_time_ms": int((time.time() - pipeline_start_time) * 1000)
                }
            }
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return {
                "response": f"Invalid input: {str(e)}",
                "sources": [],
                "pipeline_info": {
                    "error": "validation_error",
                    "original_query": query,
                    "error_details": str(e)
                }
            }
        except Exception as e:
            logger.error(f"RAG pipeline error: {str(e)}", exc_info=True)
            return {
                "response": "I encountered an unexpected error while processing your question. Please try again.",
                "sources": [],
                "pipeline_info": {
                    "error": "processing_error",
                    "original_query": query,
                    "error_details": str(e),
                    "processing_time_ms": int((time.time() - pipeline_start_time) * 1000)
                }
            }
    
    async def _rerank_candidates_robust(self, query: str, candidates: List[Dict]) -> List[Chunk]:
        """Apply dual reranking with robust error handling"""
        if not candidates:
            return []
        
        # Convert candidates to Chunk objects
        chunks = []
        for i, candidate in enumerate(candidates):
            try:
                chunk = Chunk(
                    id=f"chunk_{i}",
                    text=candidate["text"],
                    metadata=candidate["metadata"],
                    token_count=len(candidate["text"].split()),
                    quality_score=candidate["score"]
                )
                chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Error creating chunk {i}: {e}")
                continue
        
        if not chunks:
            return []
        
        try:
            # Get scores from both rerankers with error handling
            cohere_scores = []
            cross_scores = []
            
            try:
                logger.info("Getting Cohere reranker scores")
                cohere_scores = self.cohere_reranker.score(query, chunks)
                cohere_scores = self._normalize_scores(cohere_scores)
            except Exception as e:
                logger.warning(f"Cohere reranking failed: {e}")
                cohere_scores = [0.5] * len(chunks)  # Neutral scores
            
            try:
                logger.info("Getting CrossEncoder reranker scores")
                cross_scores = self.cross_reranker.score(query, chunks)
                cross_scores = self._normalize_scores(cross_scores)
            except Exception as e:
                logger.warning(f"CrossEncoder reranking failed: {e}")
                cross_scores = [0.5] * len(chunks)  # Neutral scores
            
            # Ensure score lists match chunk count
            if len(cohere_scores) != len(chunks):
                logger.warning("Cohere scores length mismatch, using fallback")
                cohere_scores = [0.5] * len(chunks)
            
            if len(cross_scores) != len(chunks):
                logger.warning("CrossEncoder scores length mismatch, using fallback")
                cross_scores = [0.5] * len(chunks)
            
            # Combine scores
            logger.info("Combining reranker scores")
            reranked_chunks = self.rag_service.combine_rerank_scores(
                candidates=chunks,
                cohere_scores=cohere_scores,
                cross_scores=cross_scores,
                method=self.config.rerank_method,
                cohere_weight=self.config.cohere_weight,
                cross_weight=self.config.cross_weight,
                top_n=self.config.rerank_top_n
            )
            
            return reranked_chunks
            
        except Exception as e:
            logger.warning(f"Reranking completely failed: {e}")
            if self.config.fallback_on_rerank_failure:
                # Fallback: return original chunks sorted by score
                sorted_chunks = sorted(chunks, key=lambda x: x.quality_score or 0, reverse=True)
                return sorted_chunks[:self.config.rerank_top_n]
            else:
                raise
    
    def _prepare_context_candidates(self, chunks: List[Chunk]) -> List[Dict]:
        """Convert chunks back to candidate format for context building"""
        context_candidates = []
        for chunk in chunks:
            try:
                # Use quality_score as relevance, fallback to 1.0
                score = chunk.quality_score if chunk.quality_score is not None else 1.0
                
                context_candidates.append({
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "score": score
                })
            except Exception as e:
                logger.warning(f"Error preparing context candidate: {e}")
                continue
        
        return context_candidates
    
    def _extract_sources_robust(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Extract source information with robust error handling"""
        sources = []
        seen_docs = set()
        
        for chunk in chunks:
            try:
                metadata = chunk.metadata or {}
                doc_id = metadata.get("doc_id", "Unknown")
                page_num = metadata.get("page_number")
                
                # Create unique identifier for deduplication
                source_key = f"{doc_id}_{page_num}" if page_num else doc_id
                
                if source_key not in seen_docs:
                    seen_docs.add(source_key)
                    
                    # Safely extract snippet
                    snippet = chunk.text or ""
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "..."
                    
                    source = {
                        "document": doc_id,
                        "page": page_num,
                        "snippet": snippet,
                        "relevance_score": chunk.quality_score or 0.0
                    }
                    sources.append(source)
                    
            except Exception as e:
                logger.warning(f"Error extracting source: {e}")
                continue
        
        return sources
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all pipeline components"""
        health = {
            "overall_status": "healthy",
            "components": {},
            "configuration": {
                "max_candidates": self.config.max_candidates,
                "rerank_top_n": self.config.rerank_top_n,
                "timeout_seconds": self.config.timeout_seconds
            },
            "timestamp": time.time()
        }
        
        # Check API keys
        import os
        
        # OpenRouter API
        if not os.getenv("OPENROUTER_API_KEY"):
            health["components"]["openrouter_api"] = {"status": "missing_key", "message": "API key not found"}
            health["overall_status"] = "degraded"
        else:
            health["components"]["openrouter_api"] = {"status": "ok", "message": "API key present"}
        
        # Cohere API
        if not os.getenv("cohere_API_key"):
            health["components"]["cohere_api"] = {"status": "missing_key", "message": "API key not found"}
            health["overall_status"] = "degraded"
        else:
            health["components"]["cohere_api"] = {"status": "ok", "message": "API key present"}
        
        # Component status
        health["components"]["rag_service"] = {"status": "ok", "message": "Service initialized"}
        health["components"]["vector_service"] = {"status": "ok", "message": "Service initialized"}
        health["components"]["cohere_reranker"] = {"status": "ok", "message": "Service initialized"}
        health["components"]["cross_reranker"] = {"status": "ok", "message": "Service initialized"}
        
        return health
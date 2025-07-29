import os
import logging
import aiohttp
import requests
from typing import List, Optional
from fastapi import WebSocket
from backend.app.models.chunk_models import Chunk

logger = logging.getLogger(__name__)

class RAGService:
    """Complete RAG service with all pipeline components integrated"""
    
    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set in environment variables.")
        logger.info("RAGService initialized successfully")

    async def enhance_query(
        self, query: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Enhance a user query using DeepSeek V3 via OpenRouter API.
        Supports synonym substitution, typo correction, and normalization.
        Falls back to original query on API failure.
        """
        if not self.api_key:
            return query
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "deepseek-chat-v3.0",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                    or "Rewrite the user query for search: correct typos, expand synonyms, normalize, but preserve intent.",
                },
                {"role": "user", "content": query},
            ],
            "max_tokens": 128,
            "temperature": 0.2,
        }
        logger.info(f"Enhancing query: {query[:50]}...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, headers=headers, json=payload, timeout=10
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        rewritten = data["choices"][0]["message"]["content"].strip()
                        logger.info(f"Query enhanced successfully: {rewritten[:50]}...")
                        return rewritten or query
                    else:
                        logger.warning(f"OpenRouter API error: {resp.status}")
        except Exception as e:
            logger.warning(f"enhance_query API call failed: {e}")
        logger.info("Returning original query due to enhancement failure")
        return query

    async def retrieve_candidates(
        self,
        query: str,
        collection_name: str,
        k: int = 20,
        metadata_filter: Optional[dict] = None,
        vector_service: Optional[object] = None,
    ) -> list:
        """
        Retrieve top-K candidates using ChromaDB vector search with optional metadata filtering.
        Deduplicate by chunk_id/hash and rank by similarity.
        """
        logger.info(f"Retrieving candidates for query: {query[:50]}... from collection: {collection_name}")
        
        if vector_service is None:
            from .vector_service import VectorService
            vector_service = VectorService()
            logger.debug("Created new VectorService instance")

        # Vector similarity search with optional metadata filter
        logger.debug(f"Performing similarity search with k={k}, metadata_filter={metadata_filter}")
        results = await vector_service.similarity_search(
            query=query,
            collection_name=collection_name,
            k=k,
            metadata_filter=metadata_filter,
        )
        logger.info(f"Retrieved {len(results)} raw candidates")
        
        # Deduplicate by chunk hash/id
        seen = set()
        deduped = []
        for r in sorted(results, key=lambda x: x.get("score", float("inf"))):
            meta = r.get("metadata", {})
            chunk_id = meta.get("hash") or meta.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                deduped.append(r)
        
        logger.info(f"After deduplication: {len(deduped)} unique candidates")
        return deduped

    async def build_context(
        self,
        candidates: list,
        max_tokens: int = 4000,
        tokenizer=None,
    ) -> str:
        """
        Build a context window from reranked candidates, fitting within max_tokens.
        Includes highest-ranked, deduplicated, and logically ordered chunks.
        """
        logger.info(f"Building context from {len(candidates)} candidates with max_tokens={max_tokens}")
        
        # Use tiktoken for consistency
        if tokenizer is None:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                tokenizer = enc
                logger.debug("Using tiktoken tokenizer")
            except ImportError:
                # Fallback to word count
                tokenizer = type("DummyTokenizer", (), {"encode": staticmethod(lambda x: x.split())})()
                logger.warning("tiktoken not available, using word count fallback")

        # Deduplicate by chunk hash/id
        seen = set()
        deduped = []
        for c in candidates:
            meta = c.get("metadata", {})
            chunk_id = meta.get("hash") or meta.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                deduped.append(c)

        # Sort by score first (higher score = higher relevance), then by source location
        def sort_key(c):
            meta = c.get("metadata", {})
            score = c.get("score", 0)
            return (
                -score,  # Negative for descending order (higher score first)
                meta.get("doc_id", ""),
                meta.get("page_number", 0),
                meta.get("section_title", ""),
            )

        ordered = sorted(deduped, key=sort_key)

        # Add chunks until token limit is reached
        context_chunks = []
        total_tokens = 0
        for c in ordered:
            chunk_text = c.get("text", "")
            meta = c.get("metadata", {})
            
            # Format with source information
            doc_name = meta.get("doc_id", "Unknown")
            page_num = meta.get("page_number")
            source_info = f"[Document: {doc_name}"
            if page_num:
                source_info += f", Page: {page_num}"
            source_info += "]"
            
            formatted_chunk = f"{source_info}\n{chunk_text}"
            tokens = tokenizer.encode(formatted_chunk)
            if total_tokens + len(tokens) > max_tokens:
                break
            context_chunks.append(formatted_chunk)
            total_tokens += len(tokens)
        
        context = "\n\n".join(context_chunks)
        logger.info(f"Built context with {len(context_chunks)} chunks, {total_tokens} tokens, {len(context)} characters")
        return context

    def combine_rerank_scores(
        self,
        candidates: List[Chunk],
        cohere_scores: List[float],
        cross_scores: List[float],
        method: str = "weighted",
        cohere_weight: float = 0.6,
        cross_weight: float = 0.4,
        top_n: int = None
    ) -> List[Chunk]:
        """
        Combine reranker scores using weighted average or ensemble voting.
        Returns top-ranked chunks sorted by fused score (descending).
        """
        logger.info(f"Combining rerank scores for {len(candidates)} candidates using {method} method")
        
        if not (len(candidates) == len(cohere_scores) == len(cross_scores)):
            logger.error(f"Length mismatch: candidates={len(candidates)}, cohere={len(cohere_scores)}, cross={len(cross_scores)}")
            raise ValueError("Candidates and score lists must have the same length.")

        fused_scores = []
        if method == "weighted":
            logger.debug(f"Using weighted fusion: cohere_weight={cohere_weight}, cross_weight={cross_weight}")
            for c_score, x_score in zip(cohere_scores, cross_scores):
                fused = cohere_weight * c_score + cross_weight * x_score
                fused_scores.append(fused)
        elif method == "vote":
            # Simple ensemble: 1 if either score above threshold, else 0
            threshold = 0.5
            for c_score, x_score in zip(cohere_scores, cross_scores):
                vote = int(c_score > threshold) + int(x_score > threshold)
                fused_scores.append(vote)
        else:
            raise ValueError("Unknown fusion method: choose 'weighted' or 'vote'.")

        # Pair chunks with scores and sort
        ranked = sorted(
            zip(candidates, fused_scores),
            key=lambda x: x[1],
            reverse=True
        )
        if top_n is not None:
            ranked = ranked[:top_n]
            logger.debug(f"Limited results to top {top_n} chunks")
        
        result_chunks = [chunk for chunk, score in ranked]
        logger.info(f"Reranking complete: returning {len(result_chunks)} chunks")
        return result_chunks

    async def generate_response(
        self,
        query: str,
        context: str,
        rag_prompt_template: str,
        stream: bool = False,
        websocket: Optional[WebSocket] = None,
        model: str = "deepseek-chat",
        api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    ) -> str:
        """
        Generate a response using DeepSeek V3 via OpenRouter API.
        Supports batch and FastAPI WebSocket streaming.
        """
        logger.info(f"Generating response for query: {query[:50]}... (stream={stream})")
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OpenRouter API key not found")
            raise ValueError("OpenRouter API key not found in environment variable 'OPENROUTER_API_KEY'")

        prompt = rag_prompt_template.format(query=query, context=context)
        logger.debug(f"Generated prompt length: {len(prompt)} characters")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant using RAG."},
                {"role": "user", "content": prompt}
            ],
            "stream": stream
        }

        if stream and websocket:
            # Streamed response via WebSocket
            logger.info("Starting streaming response via WebSocket")
            with requests.post(api_url, json=payload, headers=headers, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        await websocket.send_text(line.decode("utf-8"))
            logger.info("Streaming response completed")
            return ""
        else:
            # Standard batch mode
            logger.debug("Making batch API request to OpenRouter")
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            # Extract response text from OpenRouter format
            answer = ""
            if "choices" in data and data["choices"]:
                answer = data["choices"][0]["message"]["content"]
                logger.info(f"Generated response length: {len(answer)} characters")
            else:
                logger.warning("No choices found in API response")
            return answer



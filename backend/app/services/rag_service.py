import os
import logging
import aiohttp
import requests
from typing import List, Optional
import asyncio
from fastapi import WebSocket
from backend.app.models.chunk_models import Chunk

class RAGService:
    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logging.warning("OPENROUTER_API_KEY not set in environment variables.")

    async def enhance_query(
        self, 
        query: str, 
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List] = None,
        use_context: bool = False
    ) -> str:
        """
        Enhance a user query using DeepSeek V3 via OpenRouter API.
        
        Phase 1: Moderate enhancement (typo correction, synonym expansion, grammar improvement)
        Phase 2: Future context-awareness using conversation history
        
        Args:
            query: The user's original query
            system_prompt: Custom system prompt (optional)
            conversation_history: Previous conversation turns for context (future feature)
            use_context: Whether to use conversation history (future feature)
            
        Returns:
            Enhanced query string, falls back to original on API failure
        """
        if not self.api_key:
            logging.warning("OPENROUTER_API_KEY not available, returning original query")
            return query
            
        # Validate input
        if not query or not query.strip():
            return query
            
        # Use improved system prompt for moderate enhancement
        if not system_prompt:
            system_prompt = self._get_moderate_enhancement_prompt()
            
        # Future: Add context-awareness when use_context=True
        if use_context and conversation_history:
            system_prompt = self._get_context_aware_prompt(conversation_history)
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": "deepseek-chat-v3.0",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "max_tokens": 128,  # Increased for better enhancement
            "temperature": 0.1,  # Lower for more consistent results
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, headers=headers, json=payload, timeout=15
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "choices" in data and len(data["choices"]) > 0:
                            rewritten = data["choices"][0]["message"]["content"].strip()
                            # Validate that enhancement is reasonable
                            if rewritten and len(rewritten) <= len(query) * 3:  # Prevent excessive expansion
                                logging.info(f"Query enhanced: '{query}' -> '{rewritten}'")
                                return rewritten
                            else:
                                logging.warning("Enhanced query seems unreasonable, using original")
                                return query
                        else:
                            logging.warning("Invalid API response format")
                    else:
                        logging.warning(f"OpenRouter API error: {resp.status} - {await resp.text()}")
        except asyncio.TimeoutError:
            logging.warning("enhance_query API call timed out")
        except Exception as e:
            logging.warning(f"enhance_query API call failed: {e}")
            
        return query
        
    def _get_moderate_enhancement_prompt(self) -> str:
        """
        Get the system prompt for moderate query enhancement.
        Focuses on improving search effectiveness without adding extra information.
        """
        return """You are a query enhancement assistant. Your task is to improve user queries for better search results.

Rules:
1. Fix typos and grammar errors
2. Expand common abbreviations (ML -> machine learning, AI -> artificial intelligence)
3. Add relevant synonyms in parentheses when helpful for search
4. Improve sentence structure for clarity
5. NEVER add information not provided by the user
6. NEVER change the core meaning or intent
7. Keep the enhanced query concise and focused
8. If the query is already clear, make minimal changes

Examples:
- "How does ML work?" -> "How does machine learning (ML) work?"
- "Car maintainance tips" -> "Car maintenance tips"
- "Best AI models" -> "Best artificial intelligence (AI) models"

Return only the enhanced query, nothing else."""
        
    def _get_context_aware_prompt(self, conversation_history: List) -> str:
        """
        Future feature: Get system prompt for context-aware enhancement.
        Will use conversation history to resolve pronouns and maintain topic continuity.
        """
        # TODO: Implement context-aware enhancement in Phase 2
        context_summary = "\n".join([f"- {turn}" for turn in conversation_history[-3:]])  # Last 3 turns
        
        return f"""You are a context-aware query enhancement assistant. Use the conversation history to improve the current query.

Conversation context:
{context_summary}

Rules:
1. Use context to resolve pronouns (it, this, that, etc.)
2. Maintain topic continuity from previous turns
3. Fix typos, expand abbreviations, add synonyms
4. NEVER add information not in the user's query or conversation
5. NEVER change the core meaning or intent
6. Keep enhanced query concise and search-focused

Return only the enhanced query, nothing else."""

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
        if vector_service is None:
            from .vector_service import VectorService

            vector_service = VectorService()

        # Vector similarity search with optional metadata filter
        results = await vector_service.similarity_search(
            query=query,
            collection_name=collection_name,
            k=k,
            metadata_filter=metadata_filter,
        )
        # Deduplicate by chunk hash/id (assume 'hash' in metadata)
        seen = set()
        deduped = []
        for r in sorted(results, key=lambda x: x.get("score", float("inf"))):
            meta = r.get("metadata", {})
            chunk_id = meta.get("hash") or meta.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                deduped.append(r)
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
        # Use a default tokenizer if not provided
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            except ImportError:

                def tokenizer(text):
                    return text.split()

                tokenizer = type(
                    "DummyTokenizer", (), {"encode": staticmethod(lambda x: x.split())}
                )()

        # Deduplicate by chunk hash/id
        seen = set()
        deduped = []
        for c in candidates:
            meta = c.get("metadata", {})
            chunk_id = meta.get("hash") or meta.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                deduped.append(c)

        # Sort by source location if available, else by score
        def sort_key(c):
            meta = c.get("metadata", {})
            return (
                meta.get("doc_id", ""),
                meta.get("page_number", 0),
                meta.get("section_title", ""),
                c.get("score", float("inf")),
            )

        ordered = sorted(deduped, key=sort_key)

        # Add chunks until token limit is reached
        context_chunks = []
        total_tokens = 0
        for c in ordered:
            chunk_text = c.get("text", "")
            tokens = tokenizer.encode(chunk_text)
            if total_tokens + len(tokens) > max_tokens:
                break
            context_chunks.append(chunk_text)
            total_tokens += len(tokens)
        return "\n\n".join(context_chunks)

"""
RAGService: Orchestrates the retrieval-augmented generation pipeline, including retrieval, reranking, and LLM response generation.
"""

class RAGService:
    # ...other methods...

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
        if not (len(candidates) == len(cohere_scores) == len(cross_scores)):
            raise ValueError("Candidates and score lists must have the same length.")

        fused_scores = []
        if method == "weighted":
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
        # Return only the chunks, sorted
        return [chunk for chunk, score in ranked]

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
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not found in environment variable 'OPENROUTER_API_KEY'")

        prompt = rag_prompt_template.format(query=query, context=context)
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
            with requests.post(api_url, json=payload, headers=headers, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        await websocket.send_text(line.decode("utf-8"))
            return ""
        else:
            # Standard batch mode
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            # Extract response text from OpenRouter format
            answer = ""
            if "choices" in data and data["choices"]:
                answer = data["choices"][0]["message"]["content"]
            return answer

    # ...other methods...

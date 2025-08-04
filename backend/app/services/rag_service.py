import os
import logging
import aiohttp
import requests
from typing import List, Optional
from fastapi import WebSocket
try:
    from backend.app.models.chunk_models import Chunk
except ImportError:
    from app.models.chunk_models import Chunk

class RAGService:
    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logging.warning("OPENROUTER_API_KEY not set in environment variables.")

    async def enhance_query(
        self, query: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Enhance a user query using DeepSeek V3 via OpenRouter API.
        Supports synonym substitution, typo correction, and normalization.
        Falls back to original query on API failure.
        """
        if not self.api_key:
            logging.warning("No OpenRouter API key found. Returning original query.")
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
            "max_tokens": 64,
            "temperature": 0.2,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, headers=headers, json=payload, timeout=10
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        rewritten = data["choices"][0]["message"]["content"].strip()
                        if not rewritten:
                            logging.warning("DeepSeek V3 returned empty response. Using original query.")
                            return query
                        return rewritten
                    else:
                        logging.warning(f"OpenRouter API error: {resp.status}")
        except Exception as e:
            logging.warning(f"enhance_query API call failed: {e}")
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
        Supports both sentence-transformers and Jina embeddings.
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

    # ...other
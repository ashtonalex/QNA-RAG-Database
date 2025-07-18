import os
import logging
import aiohttp
from typing import Optional


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
                        return rewritten or query
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

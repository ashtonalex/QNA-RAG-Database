import os
import asyncio
import logging
import hashlib
import json
import aioredis
from dotenv import load_dotenv
import re
import requests
from typing import List
from sentence_transformers import SentenceTransformer


class VectorService:
    _embedding_model_warmed = False

    async def _get_redis(self):
        if not hasattr(self, "_redis"):
            load_dotenv()
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self._redis = await aioredis.from_url(
                redis_url, encoding="utf-8", decode_responses=True
            )
        return self._redis

    async def _cache_get(self, key):
        redis = await self._get_redis()
        return await redis.get(key)

    async def _cache_set(self, key, value, expire=3600):
        redis = await self._get_redis()
        await redis.set(key, value, ex=expire)

    async def _warm_embedding_model(self):
        if not self._embedding_model_warmed:
            # Pre-warm by running a dummy embedding
            try:
                await self.generate_embeddings(["warmup"])
                self._embedding_model_warmed = True
            except Exception as e:
                logging.warning(f"Embedding model warmup failed: {e}")

    async def similarity_search(
        self,
        query: str,
        collection_name: str,
        k: int = 10,
        metadata_filter: dict = None,
    ) -> list:
        """
        Retrieve top-K most relevant chunks from ChromaDB using vector similarity.
        Supports optional metadata filters and returns results with relevance scores.
        """
        # Generate embedding for the query using Jina API
        query_embedding = await self.generate_embeddings([query])
        if query_embedding and query_embedding[0] is not None:
            query_vec = query_embedding[0]
        else:
            return []

        # Get collection
        collection = await self.create_or_get_collection(collection_name)

        # Build filter
        chroma_filter = metadata_filter if metadata_filter else None

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=chroma_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results with relevance scores (lower distance = higher relevance)
        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        scores = results.get("distances", [[]])[0]
        for doc, meta, score in zip(docs, metas, scores):
            output.append({"text": doc, "metadata": meta, "score": score})
        return output

    async def generate_embeddings(
        self, chunks: list, batch_size: int = 32, max_retries: int = 3
    ) -> list:
        """
        Generate embeddings for a list of text chunks using Jina Embeddings v4 API.
        Uses async batching and caches repeated requests in Redis.
        """
        await self._warm_embedding_model()
        results = [None] * len(chunks)
        headers = {
            "Authorization": f"Bearer {self.jina_api_key}",
            "Content-Type": "application/json",
        }
        semaphore = asyncio.Semaphore(8)  # Limit concurrent batches for memory safety

        async def fetch_batch(valid_indices, valid_texts):
            # Cache key based on hash of texts
            key = (
                "jinaemb:"
                + hashlib.sha256(
                    json.dumps(valid_texts, sort_keys=True).encode()
                ).hexdigest()
            )
            cached = await self._cache_get(key)
            if cached:
                try:
                    embeddings = json.loads(cached)
                    for idx, emb in zip(valid_indices, embeddings):
                        results[idx] = emb
                    return
                except Exception:
                    pass
            for attempt in range(max_retries):
                try:

                    def fetch_embeddings():
                        payload = {"input": list(valid_texts), "model": self.jina_model}
                        response = requests.post(
                            self.jina_api_url, headers=headers, json=payload, timeout=30
                        )
                        response.raise_for_status()
                        return response.json()["data"]

                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(None, fetch_embeddings)
                    emb_vectors = [emb["embedding"] for emb in embeddings]
                    for idx, emb in zip(valid_indices, emb_vectors):
                        results[idx] = emb
                    await self._cache_set(key, json.dumps(emb_vectors))
                    break
                except Exception:
                    if attempt == max_retries - 1:
                        for idx in valid_indices:
                            results[idx] = None

        tasks = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            indices = list(range(start, start + len(batch)))
            valid = [
                (i, t)
                for i, t in zip(indices, batch)
                if isinstance(t, str) and t.strip()
            ]
            if not valid:
                continue
            valid_indices, valid_texts = zip(*valid)

            async def batch_task(valid_indices=valid_indices, valid_texts=valid_texts):
                async with semaphore:
                    await fetch_batch(valid_indices, valid_texts)

            tasks.append(batch_task())
        await asyncio.gather(*tasks)
        return results
        # The following code block was removed due to unexpected indentation and being outside any function or class.
        # If this logic is needed, please place it inside the appropriate method.

        pass


# Validation utility for embedding shape and dimensionality
def validate_embeddings(embeddings: list, expected_dim: int = 768) -> bool:
    """
    Validate that all embeddings are lists of floats with the expected dimensionality.
    Logs errors or inconsistencies for debugging and analytics.
    Returns True if all embeddings are valid, False otherwise.
    """
    all_valid = True
    for idx, emb in enumerate(embeddings):
        if emb is None:
            logging.error(f"Embedding at index {idx} is None.")
            all_valid = False
            continue
        if not isinstance(emb, list):
            logging.error(f"Embedding at index {idx} is not a list: {type(emb)}")
            all_valid = False
            continue
        if len(emb) != expected_dim:
            logging.error(
                f"Embedding at index {idx} has wrong dimension: {len(emb)} (expected {expected_dim})"
            )
            all_valid = False
            continue
        if not all(isinstance(x, (float, int)) for x in emb):
            logging.error(f"Embedding at index {idx} contains non-numeric values.")
            all_valid = False
    return all_valid

    async def generate_embeddings(
        self, chunks: list, batch_size: int = 32, max_retries: int = 3
    ) -> list:
        """
        Generates embeddings for a list of text chunks using jina-embeddings-v2-small-en.
        Supports batch processing, retries on failure, and handles empty/malformed chunks.
        Returns a list of embedding vectors (or None for failed chunks).
        """
        from jina import JinaEmbeddings
        import asyncio
        import logging

        if not hasattr(self, "embedding_model"):
            self.embedding_model = JinaEmbeddings(
                model_name="jina-embeddings-v2-small-en"
            )

        results = [None] * len(chunks)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            indices = list(range(start, start + len(batch)))
            # Filter out empty/malformed
            valid = [
                (i, t)
                for i, t in zip(indices, batch)
                if isinstance(t, str) and t.strip()
            ]
            if not valid:
                continue
            valid_indices, valid_texts = zip(*valid)
            for attempt in range(max_retries):
                try:
                    # JinaEmbeddings is sync, so run in thread pool
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        None, self.embedding_model.embed, list(valid_texts)
                    )
                    for idx, emb in zip(valid_indices, embeddings):
                        results[idx] = emb
                    break
                except Exception as e:
                    logging.warning(
                        f"Embedding batch failed (attempt {attempt + 1}): {e}"
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
        return results

    async def similarity_search(
        self,
        query: str,
        collection_name: str,
        k: int = 10,
        metadata_filter: dict = None,
    ) -> list:
        """
        Retrieve top-K most relevant chunks from ChromaDB using vector similarity.
        Supports optional metadata filters and returns results with relevance scores.
        Uses Redis cache for repeated queries and minimizes memory overhead.
        """
        # Cache key for query
        cache_key = f"simsearch:{collection_name}:{hashlib.sha256((query + json.dumps(metadata_filter or {})).encode()).hexdigest()}:{k}"
        cached = await self._cache_get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass

        # Generate embedding for the query using Jina API
        query_embedding = await self.generate_embeddings([query])
        if query_embedding and query_embedding[0] is not None:
            query_vec = query_embedding[0]
        else:
            return []

        # Get collection
        collection = await self.create_or_get_collection(collection_name)

        # Build filter
        chroma_filter = metadata_filter if metadata_filter else None

        # Query ChromaDB (streaming if available)
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=chroma_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results with relevance scores (lower distance = higher relevance)
        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        scores = results.get("distances", [[]])[0]
        for doc, meta, score in zip(docs, metas, scores):
            output.append({"text": doc, "metadata": meta, "score": score})
        # Cache the result
        await self._cache_set(cache_key, json.dumps(output))
        return output

    @staticmethod
    def assign_metadata(
        chunk: str, doc_id: str, section_title: str = None, page_number: int = None
    ) -> dict:
        """
        Assigns metadata to a text chunk, including doc_id, section_title, page_number, and SHA-256 hash.
        Returns a dictionary suitable for ChromaDB metadata storage.
        """
        import hashlib

        meta = {
            "doc_id": doc_id,
            "hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
        }
        if section_title is not None:
            meta["section_title"] = section_title
        if page_number is not None:
            meta["page_number"] = page_number
        return meta

    def _get_chromadb_client(self):
        import chromadb
        from chromadb.config import Settings

        return chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )

    async def create_or_get_collection(
        self, name: str, version: int = 1, scope: str = None
    ):
        """
        Initializes or retrieves a ChromaDB persistent collection for a given document or project.
        Supports versioning and document/project-scoped naming.
        Returns the collection object.
        """
        client = getattr(self, "client", None)
        if client is None:
            client = self._get_chromadb_client()
            self.client = client
        # Compose collection name
        collection_name = name
        if scope:
            collection_name = f"{scope}_{name}"
        if version:
            collection_name = f"{collection_name}_v{version}"
        # Create or get collection
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            collection = client.create_collection(collection_name)
        return collection

    def __init__(self):
        self.chunk_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def semantic_chunk(
        self, text: str, max_tokens: int = 200, overlap: int = 50
    ) -> List[str]:
        """
        Splits text into semantically meaningful chunks using a sliding window with overlap.
        Tries to preserve document structure (headings/sections) and avoid splitting mid-section.
        Each chunk is at most max_tokens tokens, with overlap tokens shared between consecutive chunks.
        """
        # Heuristic regex for headings: numbered, all-caps, or markdown style
        heading_pattern = re.compile(
            r"^(\s*(\d+\.|[A-Z][A-Z\s\-:]+|#+)\s+.+)$", re.MULTILINE
        )
        # Split text into sections by headings
        sections = []
        last_idx = 0
        for match in heading_pattern.finditer(text):
            start = match.start()
            if last_idx < start:
                section_text = text[last_idx:start].strip()
                if section_text:
                    sections.append(section_text)
            last_idx = start
        # Add the last section
        if last_idx < len(text):
            section_text = text[last_idx:].strip()
            if section_text:
                sections.append(section_text)

        # If no headings found, treat whole text as one section
        if not sections:
            sections = [text.strip()]

        tokenizer = self.chunk_model.tokenizer
        chunks = []
        for section in sections:
            # Split section into sentences
            sentences = re.split(r"(?<=[.!?])\s+", section)
            current_chunk = []
            current_length = 0
            for sentence in sentences:
                tokens = tokenizer.tokenize(sentence)
                num_tokens = len(tokens)
                if current_length + num_tokens > max_tokens:
                    # Finalize current chunk
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    # Start new chunk with overlap
                    if overlap > 0 and len(chunks) > 0:
                        overlap_tokens = []
                        overlap_count = 0
                        for s in reversed(current_chunk):
                            s_tokens = tokenizer.tokenize(s)
                            overlap_tokens = [s] + overlap_tokens
                            overlap_count += len(s_tokens)
                            if overlap_count >= overlap:
                                break
                        current_chunk = overlap_tokens.copy()
                        current_length = sum(
                            len(tokenizer.tokenize(s)) for s in current_chunk
                        )
                    else:
                        current_chunk = []
                        current_length = 0
                current_chunk.append(sentence)
                current_length += num_tokens
            # Add any remaining chunk in this section
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
        return chunks

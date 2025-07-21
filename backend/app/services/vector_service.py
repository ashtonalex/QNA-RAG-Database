import os
import asyncio
import logging
import hashlib
import json
import redis.asyncio as aioredis
from dotenv import load_dotenv
import re
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
#from jina import JinaEmbeddings

class VectorService:
    def __init__(self):
        self.chunk_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(
            "jinaai/jina-embeddings-v2-small-en", trust_remote_code=True
        )
        self.embedding_model.max_seq_length = 1024
        self._embedding_model_warmed = False

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
            try:
                await self.generate_embeddings(["warmup"])
                self._embedding_model_warmed = True
            except Exception as e:
                logging.warning(f"Embedding model warmup failed: {e}")

    async def generate_embeddings(
        self, chunks: list, batch_size: int = 32, max_retries: int = 3
    ) -> list:
        """
        Generate embeddings for a list of text chunks using local jina-embeddings-v2-small-en via sentence-transformers.
        Uses async batching and caches repeated requests in Redis.
        """
        await self._warm_embedding_model()
        results = [None] * len(chunks)
        semaphore = asyncio.Semaphore(8)  # Limit concurrent batches for memory safety

        async def fetch_batch(valid_indices, valid_texts):
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
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        None, self.embedding_model.encode, list(valid_texts)
                    )
                    embeddings = [emb.tolist() for emb in embeddings]
                    for idx, emb in zip(valid_indices, embeddings):
                        results[idx] = emb
                    await self._cache_set(key, json.dumps(embeddings))
                    break
                except Exception as e:
                    logging.warning(
                        f"Embedding batch failed (attempt {attempt + 1}): {e}"
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
            else:
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
        client = getattr(self, "client", None)
        if client is None:
            client = self._get_chromadb_client()
            self.client = client
        collection_name = name
        if scope:
            collection_name = f"{scope}_{name}"
        if version:
            collection_name = f"{collection_name}_v{version}"
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            collection = client.create_collection(collection_name)
        return collection

    @staticmethod
    def assign_metadata(
        chunk: str, doc_id: str, section_title: str = None, page_number: int = None
    ) -> dict:
        meta = {
            "doc_id": doc_id,
            "hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
        }
        if section_title is not None:
            meta["section_title"] = section_title
        if page_number is not None:
            meta["page_number"] = page_number
        return meta

    async def store_chunks(self, chunks: list, collection_name: str, doc_id: str, section_title: str = None, page_number: int = None):
        """
        Store unique chunks (deduplicated by content hash) with embeddings and metadata in ChromaDB.
        """
        # Deduplicate by content hash before storage
        seen_hashes = set()
        unique_chunks = []
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                unique_chunks.append(chunk)
        embeddings = await self.generate_embeddings(unique_chunks)
        collection = await self.create_or_get_collection(collection_name)
        metadatas = [self.assign_metadata(chunk, doc_id, section_title, page_number) for chunk in unique_chunks]
        ids = [meta["hash"] for meta in metadatas]  # Use hash as unique ID
        collection.add(
            ids=ids,
            documents=unique_chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def similarity_search(self, query: str, collection_name: str, k: int = 10, metadata_filter: dict = None) -> List[Dict]:
        """
        Retrieve top-K most relevant unique chunks from ChromaDB using vector similarity and optional metadata filter.
        Deduplicate results by content hash before returning.
        """
        query_embedding = await self.generate_embeddings([query])
        if query_embedding and query_embedding[0] is not None:
            query_vec = query_embedding[0]
        else:
            return []
        collection = await self.create_or_get_collection(collection_name)
        chroma_filter = metadata_filter if metadata_filter else None
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=k * 2,  # Fetch more to allow deduplication
            where=chroma_filter,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        scores = results.get("distances", [[]])[0]
        seen_hashes = set()
        for doc, meta, score in zip(docs, metas, scores):
            chunk_hash = meta.get("hash")
            if chunk_hash and chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                output.append({"text": doc, "metadata": meta, "score": score})
            if len(output) >= k:
                break
        return output

    async def semantic_chunk(
        self, text: str, max_tokens: int = 200, overlap: int = 50
    ) -> list:
        """
        Splits text into semantically meaningful chunks using a sliding window with overlap.
        Tries to preserve document structure (headings/sections) and avoid splitting mid-section.
        Each chunk is at most max_tokens tokens, with overlap tokens shared between consecutive chunks.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return []
        heading_pattern = re.compile(
            r"^(\s*(\d+\.|[A-Z][A-Z\s\-:]+|#+)\s+.+)$", re.MULTILINE
        )
        sections = []
        last_idx = 0
        for match in heading_pattern.finditer(text):
            start = match.start()
            if last_idx < start:
                section_text = text[last_idx:start].strip()
                if section_text:
                    sections.append(section_text)
            last_idx = start
        if last_idx < len(text):
            section_text = text[last_idx:].strip()
            if section_text:
                sections.append(section_text)
        if not sections:
            sections = [text.strip()]
        tokenizer = self.chunk_model.tokenizer
        chunks = []
        for section in sections:
            sentences = re.split(r"(?<=[.!?])\s+", section)
            current_chunk = []
            current_length = 0
            for sentence in sentences:
                tokens = tokenizer.tokenize(sentence)
                num_tokens = len(tokens)
                if current_length + num_tokens > max_tokens:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
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
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
        return chunks

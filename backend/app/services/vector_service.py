import re
from typing import List
from sentence_transformers import SentenceTransformer


class VectorService:
    async def store_chunks(
        self, chunks: list, collection_name: str, batch_size: int = 100
    ):
        """
        Batch-inserts chunks (with metadata) into the specified ChromaDB collection.
        Deduplicates by hash and uses batch writing for high throughput.
        """
        # Get or create collection
        collection = await self.create_or_get_collection(collection_name)
        # Fetch existing hashes for deduplication
        existing_hashes = set()
        try:
            # ChromaDB: get all hashes in collection (may need to page for large collections)
            offset = 0
            limit = 1000
            while True:
                results = collection.get(
                    where={"hash": {"$exists": True}},
                    limit=limit,
                    offset=offset,
                    include=["metadatas"],
                )
                if not results or not results.get("metadatas"):
                    break
                for meta in results["metadatas"]:
                    if meta and "hash" in meta:
                        existing_hashes.add(meta["hash"])
                if len(results["metadatas"]) < limit:
                    break
                offset += limit
        except Exception:
            pass  # If collection is empty or get fails, just proceed

        # Prepare new chunks (deduplicate by hash)
        new_chunks = []
        new_metadatas = []
        ids = []
        for chunk in chunks:
            text = chunk["text"] if isinstance(chunk, dict) else chunk
            meta = (
                chunk["metadata"]
                if isinstance(chunk, dict) and "metadata" in chunk
                else {}
            )
            chunk_hash = meta.get("hash")
            if not chunk_hash:
                import hashlib

                chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                meta["hash"] = chunk_hash
            if chunk_hash in existing_hashes:
                continue  # skip duplicate
            ids.append(chunk_hash)
            new_chunks.append(text)
            new_metadatas.append(meta)

        # Batch insert
        for i in range(0, len(new_chunks), batch_size):
            batch_texts = new_chunks[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metas = new_metadatas[i : i + batch_size]
            if batch_texts:
                collection.add(
                    documents=batch_texts, metadatas=batch_metas, ids=batch_ids
                )

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

import logging
from typing import Dict, Optional
from .vector_service import VectorService


class VectorPipelineService:
    def __init__(self):
        self.vector_service = VectorService()

    async def process_and_store(
        self,
        text: str,
        doc_id: str,
        section_title: Optional[str] = None,
        page_number: Optional[int] = None,
        collection_name: str = "default_collection",
        batch_size: int = 32,
    ) -> Dict:
        """
        Full pipeline: chunk -> embed -> store in ChromaDB.
        Returns dict with status, chunk count, and stored IDs.
        """
        try:
            # 1. Semantic chunking
            logging.info("Starting semantic chunking...")
            chunks = await self.vector_service.semantic_chunk(text)
            if not chunks:
                logging.error("No chunks produced from input text.")
                return {"status": "error", "reason": "No chunks produced"}

            # 2. Assign metadata to each chunk
            metadatas = [
                self.vector_service.assign_metadata(
                    chunk, doc_id, section_title, page_number
                )
                for chunk in chunks
            ]

            # 3. Embedding generation (async batch)
            logging.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = await self.vector_service.generate_embeddings(
                chunks, batch_size=batch_size
            )
            if not embeddings or len(embeddings) != len(chunks):
                logging.error("Embedding count does not match chunk count.")
                return {"status": "error", "reason": "Embedding failure"}

            # 4. Store in ChromaDB (async batch)
            logging.info("Storing chunks, embeddings, and metadata in ChromaDB...")
            await self.vector_service.insert_chunks_with_embeddings(
                chunks, embeddings, metadatas, collection_name, batch_size=batch_size
            )

            # 5. Return confirmation and chunk IDs
            chunk_ids = [meta["hash"] for meta in metadatas]
            logging.info(f"Pipeline complete. Stored {len(chunk_ids)} chunks.")
            return {"status": "success", "count": len(chunk_ids), "ids": chunk_ids}
        except Exception as e:
            logging.exception(f"Pipeline failed: {e}")
            return {"status": "error", "reason": str(e)}


# Optionally, you can add an API endpoint using FastAPI or similar framework.
# Example (not included here):
# from fastapi import FastAPI, Request
# app = FastAPI()
# pipeline = VectorPipelineService()
# @app.post("/process_and_store")
# async def process_and_store_endpoint(request: Request):
#     data = await request.json()
#     return await pipeline.process_and_store(**data)

# Integration test example (to be placed in a test file):
# async def test_pipeline():
#     pipeline = VectorPipelineService()
#     result = await pipeline.process_and_store("Some text...", "doc123")
#     assert result["status"] == "success"
#     assert result["count"] == len(result["ids"])

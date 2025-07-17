import pytest
import asyncio
from backend.app.models.chunk_models import ChunkingConfig
from backend.app.services.chunking_service import chunk_document, ChunkingService


@pytest.mark.asyncio
async def test_chunk_document_and_vector_payloads():
    text = (
        "This is the first sentence. Here is another one! Yet another sentence follows."
    )
    config = ChunkingConfig(chunk_size=10, overlap=0.1, strategy="hybrid")
    chunks = await chunk_document(text, config)
    assert len(chunks) > 0
    # Test output format for vector DB
    payloads = ChunkingService(config).chunks_to_vector_payloads(chunks)
    assert isinstance(payloads, list)
    assert all("id" in p and "text" in p and "metadata" in p for p in payloads)
    # Test with a different config
    config2 = ChunkingConfig(chunk_size=2, overlap=0.0, strategy="syntactic")
    chunks2 = await chunk_document(text, config2)
    payloads2 = ChunkingService(config2).chunks_to_vector_payloads(chunks2)
    assert len(payloads2) > 0
    assert all(isinstance(p["text"], str) for p in payloads2)


@pytest.mark.asyncio
async def test_chunk_document_metadata_propagation():
    text = (
        "Section 1: Introduction. This is the first paragraph. It has two sentences.\n\n"
        "Section 2: Methods. A completely unrelated topic starts here. It is about something else entirely.\n\n"
        "Section 3: Results. Back to the original topic. This sentence is similar to the first paragraph."
    )
    config = ChunkingConfig(chunk_size=50, overlap=0.1, strategy="hybrid")
    metadata = {"page_number": 1, "section_header": "Introduction"}
    chunks = await chunk_document(text, config, metadata)
    assert len(chunks) > 0
    for chunk in chunks:
        # Metadata should be present and include propagated fields
        assert "page_number" in chunk.metadata
        assert "section_header" in chunk.metadata
        assert "index" in chunk.metadata
        assert chunk.relationships is not None
        assert "prev" in chunk.relationships and "next" in chunk.relationships
        assert isinstance(chunk.token_count, int)
        assert chunk.quality_score is not None
        assert chunk.text


# --- Manual/demo code from main branch ---
from backend.app.services.chunking_service import ChunkingService


sample_text = (
    "Section 1: Introduction. This is the first paragraph. It has two sentences.\n\n"
    "Section 1: Introduction. This is the second paragraph. It also has two sentences.\n\n"
    "Section 2: Methods. A completely unrelated topic starts here. It is about something else entirely.\n\n"
    "Section 3: Results. Back to the original topic. This sentence is similar to the first paragraph."
)


def get_mock_metadata(index):
    if index < 2:
        return {"page_number": 1, "section_header": "Introduction", "order": index + 1}
    elif index == 2:
        return {"page_number": 2, "section_header": "Methods", "order": index + 1}
    else:
        return {"page_number": 3, "section_header": "Results", "order": index + 1}


config = ChunkingConfig(chunk_size=100)
chunker = ChunkingService(config)


def print_chunk_info(chunks):
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"  Text: {chunk.text}")
        print(f"  Metadata: {chunk.metadata}")
        print(f"  Token count: {chunk.token_count}")
        print(f"  Quality score: {getattr(chunk, 'quality_score', None)}")
        print(f"  Relationships: {getattr(chunk, 'relationships', None)}")
        print()


if __name__ == "__main__":
    print("Testing Syntactic Chunking:")
    syntactic_chunks = chunker.syntactic_chunk(sample_text, metadata={})
    for i, chunk_text in enumerate(syntactic_chunks):
        print(f"Syntactic Chunk {i + 1}: {chunk_text}")
    print()

    print("Testing Semantic Chunking:")

    # semantic_chunk is now async, so we need to run it in an event loop
    async def run_semantic():
        semantic_chunks = await chunker.semantic_chunk(sample_text)
        for i, chunk_text in enumerate(semantic_chunks):
            print(f"Semantic Chunk {i + 1}: {chunk_text}")
        print()

    asyncio.run(run_semantic())

    print("Testing Hybrid Chunking (with overlap and metadata):")

    async def test_hybrid():
        # Hybrid chunking returns Chunk objects
        hybrid_chunks = await chunker.hybrid_chunk(sample_text, metadata={})
        # Attach mock metadata for demonstration
        for i, chunk in enumerate(hybrid_chunks):
            chunk.metadata.update(get_mock_metadata(i))
        print_chunk_info(hybrid_chunks)

    asyncio.run(test_hybrid())

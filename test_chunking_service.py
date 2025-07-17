import asyncio
from backend.app.models.chunk_models import ChunkingConfig
from backend.app.services.chunking_service import ChunkingService

# Sample text with mock page and section breaks
sample_text = (
    "Section 1: Introduction. This is the first paragraph. It has two sentences.\n\n"
    "Section 1: Introduction. This is the second paragraph. It also has two sentences.\n\n"
    "Section 2: Methods. A completely unrelated topic starts here. It is about something else entirely.\n\n"
    "Section 3: Results. Back to the original topic. This sentence is similar to the first paragraph."
)

# Mock metadata for demonstration (in a real pipeline, this would come from the document processor)
def get_mock_metadata(index):
    if index < 2:
        return {"page_number": 1, "section_header": "Introduction", "order": index+1}
    elif index == 2:
        return {"page_number": 2, "section_header": "Methods", "order": index+1}
    else:
        return {"page_number": 3, "section_header": "Results", "order": index+1}

config = ChunkingConfig(chunk_size=100)
chunker = ChunkingService(config)

def print_chunk_info(chunks):
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
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
        print(f"Syntactic Chunk {i+1}: {chunk_text}")
    print()

    print("Testing Semantic Chunking:")
    semantic_chunks = chunker.semantic_chunk(sample_text)
    for i, chunk_text in enumerate(semantic_chunks):
        print(f"Semantic Chunk {i+1}: {chunk_text}")
    print()

    print("Testing Hybrid Chunking (with overlap and metadata):")
    async def test_hybrid():
        # Hybrid chunking returns Chunk objects
        hybrid_chunks = await chunker.hybrid_chunk(sample_text, metadata={})
        # Attach mock metadata for demonstration
        for i, chunk in enumerate(hybrid_chunks):
            chunk.metadata.update(get_mock_metadata(i))
        print_chunk_info(hybrid_chunks)
    asyncio.run(test_hybrid())
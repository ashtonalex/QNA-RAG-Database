from typing import List, Dict
from backend.app.models.chunk_models import Chunk, ChunkingConfig
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

class ChunkingService:
    def __init__(self, config: ChunkingConfig):
        """
        Initialize the chunking service with a given configuration.
        """
        self.config = config
        # Ensure NLTK punkt is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        # Load sentence-transformer model for semantic chunking
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # TODO: Initialize tokenizer and other resources based on config

    async def hybrid_chunk(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Perform hybrid chunking (syntactic first, then semantic grouping, then overlap).
        Returns a list of Chunk objects.
        """
        if not text or not isinstance(text, str):
            return []
        # Step 1: Syntactic chunking (paragraphs/sentences)
        syntactic_chunks = self.syntactic_chunk(text, metadata)
        # Step 2: Semantic grouping within syntactic chunks
        semantic_chunks = []
        for chunk in syntactic_chunks:
            # Each syntactic chunk may contain multiple sentences
            grouped = self.semantic_chunk(chunk)
            semantic_chunks.extend(grouped)
        # Step 3: Overlap logic
        overlapped_chunks = self.create_overlapping_chunks(semantic_chunks)
        return overlapped_chunks

    def syntactic_chunk(self, text: str, metadata: Dict) -> List[str]:
        """
        Split text into syntactic chunks (paragraphs, then sentences).
        Returns a list of text chunks.
        """
        if not text or not isinstance(text, str):
            return []
        # Split by double newlines for paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        for para in paragraphs:
            # Further split paragraphs into sentences if chunk_size is small
            if self.config.chunk_size < 200:  # Arbitrary threshold for demo
                sentences = sent_tokenize(para)
                chunks.extend([s for s in sentences if s.strip()])
            else:
                chunks.append(para)
        return chunks

    def semantic_chunk(self, text: str) -> List[str]:
        """
        Group semantically similar sentences into chunks.
        Returns a list of text chunks.
        """
        if not text or not isinstance(text, str):
            return []
        # Split text into sentences
        sentences = sent_tokenize(text, language='english')
        if not sentences:
            return []
        # Embed all sentences
        embeddings = self.model.encode(sentences)
        # Group sentences into chunks based on semantic similarity
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        for i in range(1, len(sentences)):
            sim = cosine_similarity([current_embedding], [embeddings[i]])[0][0]
            # If similarity drops below threshold or chunk is too large, start new chunk
            if sim < 0.7 or len(' '.join(current_chunk).split()) > self.config.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]
            else:
                current_chunk.append(sentences[i])
                # Update current embedding as mean of chunk
                current_embedding = np.mean(embeddings[i-len(current_chunk)+1:i+1], axis=0)
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken for OpenAI compatibility.
        """
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Fallback to word count if tiktoken fails
            return len(text.split())

    def _score_chunk(self, token_count: int) -> float:
        """
        Simple quality score: 1.0 if within ideal range, less if too short/long.
        """
        ideal = self.config.chunk_size
        min_tokens = int(0.5 * ideal)
        max_tokens = int(1.5 * ideal)
        if min_tokens <= token_count <= max_tokens:
            return 1.0
        # Linear penalty for deviation
        if token_count < min_tokens:
            return max(0.0, token_count / min_tokens)
        else:
            return max(0.0, (max_tokens - (token_count - max_tokens)) / max_tokens)

    def create_overlapping_chunks(self, chunks: List[str]) -> List[Chunk]:
        """
        Add overlap between chunks for context preservation.
        Returns a list of Chunk objects with overlap.
        """
        if not chunks:
            return []
        overlap_size = max(1, int(self.config.overlap * self.config.chunk_size))
        overlapped_chunks = []
        for i, chunk_text in enumerate(chunks):
            # Get previous chunk's tail for overlap
            if i > 0 and overlap_size > 0:
                prev_chunk_words = overlapped_chunks[-1].text.split()
                overlap_words = prev_chunk_words[-overlap_size:] if len(prev_chunk_words) >= overlap_size else prev_chunk_words
                chunk_text = ' '.join(overlap_words) + ' ' + chunk_text
            token_count = self._count_tokens(chunk_text)
            quality_score = self._score_chunk(token_count)
            chunk = Chunk(
                id=f"chunk_{i+1}",
                text=chunk_text,
                metadata={"index": i+1},
                token_count=token_count,
                quality_score=quality_score,
                relationships={
                    "prev": f"chunk_{i}" if i > 0 else None,
                    "next": f"chunk_{i+2}" if i < len(chunks) - 1 else None
                }
            )
            overlapped_chunks.append(chunk)
        return overlapped_chunks

    # Additional utility methods (e.g., token counting, quality scoring) can be added here.

# Simple test for syntactic_chunk
if __name__ == "__main__":
    import asyncio
    from backend.app.models.chunk_models import ChunkingConfig
    sample_text = (
        "This is the first paragraph. It has two sentences.\n\n"
        "This is the second paragraph. It also has two sentences.\n\n"
        "A completely unrelated topic starts here. It is about something else entirely.\n\n"
        "Back to the original topic. This sentence is similar to the first paragraph."
    )
    config = ChunkingConfig(chunk_size=100)
    chunker = ChunkingService(config)
    print("Syntactic Chunks:")
    syntactic_chunks = chunker.syntactic_chunk(sample_text, metadata={})
    for i, chunk in enumerate(syntactic_chunks):
        print(f"Chunk {i+1}: {chunk}")
    print("\nSemantic Chunks:")
    semantic_chunks = chunker.semantic_chunk(sample_text)
    for i, chunk in enumerate(semantic_chunks):
        print(f"Chunk {i+1}: {chunk}")
    print("\nHybrid Chunks:")
    async def test_hybrid():
        hybrid_chunks = await chunker.hybrid_chunk(sample_text, metadata={})
        for i, chunk in enumerate(hybrid_chunks):
            if hasattr(chunk, 'text') and hasattr(chunk, 'token_count'):
                print(f"Chunk {i+1}: {chunk.text} (tokens: {chunk.token_count}, score: {chunk.quality_score:.2f})")
            elif hasattr(chunk, 'text'):
                print(f"Chunk {i+1}: {chunk.text}")
            else:
                print(f"Chunk {i+1}: {chunk}")
    asyncio.run(test_hybrid()) 
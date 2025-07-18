import re
from typing import List
from sentence_transformers import SentenceTransformer


class VectorService:
    def __init__(self):
        self.chunk_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def semantic_chunk(
        self, text: str, max_tokens: int = 200, overlap: int = 50
    ) -> List[str]:
        """
        Splits text into semantically meaningful chunks using a sliding window with overlap.
        Each chunk is at most max_tokens tokens, with overlap tokens shared between consecutive chunks.
        """
        # Split text into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        # Tokenize sentences using the model's tokenizer
        tokenizer = self.chunk_model.tokenizer
        chunks = []
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
                    # Add sentences from the end until overlap is reached
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
        # Add any remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
        return chunks

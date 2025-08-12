"""
CrossEncoderReranker: Uses a sentence-transformers cross-encoder model for semantic reranking of document chunks.
"""

from typing import List
from sentence_transformers import CrossEncoder
from app.models.chunk_models import Chunk

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.cross_encoder = CrossEncoder(model_name)

    def score(self, query: str, chunks: List[Chunk], top_n: int = None) -> List[float]:
        # Optionally limit to top_n chunks
        if top_n is not None:
            chunks = chunks[:top_n]
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.cross_encoder.predict(pairs)
        # Ensure output length matches input chunk list
        if len(scores) < len(chunks):
            # Pad with zeros if needed
            scores = list(scores) + [0.0] * (len(chunks) - len(scores))
        elif len(scores) > len(chunks):
            scores = list(scores)[:len(chunks)]
        return list(scores)
"""
CohereReranker: Integrates with Cohere's Rerank API for semantic reranking of document chunks.
"""

import os
import requests
import time
from typing import List
from backend.app.models.chunk_models import Chunk

class CohereReranker:
    def __init__(self, api_key_env: str = "COHERE_API_KEY", max_retries: int = 3, backoff: float = 1.0):
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError("Cohere API key not found in environment variable '{}'".format(api_key_env))
        self.endpoint = "https://api.cohere.ai/v1/rerank"
        self.max_retries = max_retries
        self.backoff = backoff

    def score(self, query: str, chunks: List[Chunk], top_n: int = 10) -> List[float]:
        documents = [chunk.text for chunk in chunks][:top_n]
        payload = {
            "query": query,
            "documents": documents,
            "top_n": top_n
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(self.endpoint, json=payload, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                # Cohere returns a list of dicts with 'index' and 'relevance_score'
                scores = [0.0] * len(documents)
                for item in data.get("results", []):
                    idx = item.get("index")
                    score = item.get("relevance_score", 0.0)
                    if idx is not None and 0 <= idx < len(scores):
                        scores[idx] = score
                return scores
            except Exception as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Cohere Rerank API failed after {self.max_retries} attempts: {e}")
                time.sleep(self.backoff * attempt)
        return [0.0]
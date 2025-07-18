"""
RAGService: Orchestrates the retrieval-augmented generation pipeline, including retrieval, reranking, and LLM response generation.
"""

import os
import requests
from typing import List, Optional
from fastapi import WebSocket
from backend.app.models.chunk_models import Chunk

class RAGService:
    # ...other methods...

    def combine_rerank_scores(
        self,
        candidates: List[Chunk],
        cohere_scores: List[float],
        cross_scores: List[float],
        method: str = "weighted",
        cohere_weight: float = 0.6,
        cross_weight: float = 0.4,
        top_n: int = None
    ) -> List[Chunk]:
        """
        Combine reranker scores using weighted average or ensemble voting.
        Returns top-ranked chunks sorted by fused score (descending).
        """
        if not (len(candidates) == len(cohere_scores) == len(cross_scores)):
            raise ValueError("Candidates and score lists must have the same length.")

        fused_scores = []
        if method == "weighted":
            for c_score, x_score in zip(cohere_scores, cross_scores):
                fused = cohere_weight * c_score + cross_weight * x_score
                fused_scores.append(fused)
        elif method == "vote":
            # Simple ensemble: 1 if either score above threshold, else 0
            threshold = 0.5
            for c_score, x_score in zip(cohere_scores, cross_scores):
                vote = int(c_score > threshold) + int(x_score > threshold)
                fused_scores.append(vote)
        else:
            raise ValueError("Unknown fusion method: choose 'weighted' or 'vote'.")

        # Pair chunks with scores and sort
        ranked = sorted(
            zip(candidates, fused_scores),
            key=lambda x: x[1],
            reverse=True
        )
        if top_n is not None:
            ranked = ranked[:top_n]
        # Return only the chunks, sorted
        return [chunk for chunk, score in ranked]

    async def generate_response(
        self,
        query: str,
        context: str,
        rag_prompt_template: str,
        stream: bool = False,
        websocket: Optional[WebSocket] = None,
        model: str = "deepseek-chat",
        api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    ) -> str:
        """
        Generate a response using DeepSeek V3 via OpenRouter API.
        Supports batch and FastAPI WebSocket streaming.
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not found in environment variable 'OPENROUTER_API_KEY'")

        prompt = rag_prompt_template.format(query=query, context=context)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant using RAG."},
                {"role": "user", "content": prompt}
            ],
            "stream": stream
        }

        if stream and websocket:
            # Streamed response via WebSocket
            with requests.post(api_url, json=payload, headers=headers, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        await websocket.send_text(line.decode("utf-8"))
            return ""
        else:
            # Standard batch mode
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            # Extract response text from OpenRouter format
            answer = ""
            if "choices" in data and data["choices"]:
                answer = data["choices"][0]["message"]["content"]
            return answer

    # ...other methods...
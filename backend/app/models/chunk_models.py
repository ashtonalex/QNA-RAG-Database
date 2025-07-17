from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class Chunk(BaseModel):
    id: str
    text: str
    metadata: Dict = Field(default_factory=dict)
    token_count: int
    embedding: Optional[List[float]] = None
    quality_score: Optional[float] = None
    relationships: Optional[Dict] = None  # e.g., {'prev': chunk_id, 'next': chunk_id}
    source_doc_id: Optional[str] = None

class ChunkingConfig(BaseModel):
    chunk_size: int = 500
    overlap: float = 0.1  # 10% overlap by default
    strategy: str = "hybrid"  # 'syntactic', 'semantic', or 'hybrid'
    language: Optional[str] = "en"
    tokenizer: Optional[str] = "tiktoken"  # or 'spacy', 'hf', etc. 
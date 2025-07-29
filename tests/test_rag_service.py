import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from backend.app.services.rag_service import RAGService

@pytest.mark.asyncio
async def test_enhance_query_success():
    service = RAGService()
    mock_response = {
        "choices": [
            {"message": {"content": "expanded and normalized query"}}
        ]
    }
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        result = await service.enhance_query("orig query")
        assert result == "expanded and normalized query"

@pytest.mark.asyncio
async def test_enhance_query_api_failure():
    service = RAGService()
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.side_effect = Exception("API error")
        result = await service.enhance_query("orig query")
        assert result == "orig query"

@pytest.mark.asyncio
async def test_retrieve_candidates_dedup_and_filter():
    service = RAGService()
    mock_chunks = [
        {"text": "A", "score": 0.1, "metadata": {"hash": "1", "section": "X"}},
        {"text": "B", "score": 0.2, "metadata": {"hash": "2", "section": "Y"}},
        {"text": "C", "score": 0.3, "metadata": {"hash": "1", "section": "X"}},  # duplicate hash
    ]
    mock_vector_service = AsyncMock()
    mock_vector_service.similarity_search.return_value = mock_chunks

    result = await service.retrieve_candidates(
        query="test",
        collection_name="test_collection",
        k=10,
        metadata_filter={"section": "X"},
        vector_service=mock_vector_service,
    )
    # Only one chunk with hash "1" should remain, and metadata filter should apply
    assert any(c["metadata"]["hash"] == "1" for c in result)
    assert all(c["metadata"]["section"] == "X" for c in result)
    assert len(result) == 1

@pytest.mark.asyncio
async def test_retrieve_candidates_no_duplicates():
    service = RAGService()
    mock_chunks = [
        {"text": "A", "score": 0.1, "metadata": {"hash": "1"}},
        {"text": "B", "score": 0.2, "metadata": {"hash": "2"}},
        {"text": "C", "score": 0.3, "metadata": {"hash": "3"}},
    ]
    mock_vector_service = AsyncMock()
    mock_vector_service.similarity_search.return_value = mock_chunks

    result = await service.retrieve_candidates(
        query="test",
        collection_name="test_collection",
        k=10,
        metadata_filter=None,
        vector_service=mock_vector_service,
    )
    hashes = [c["metadata"]["hash"] for c in result]
    assert hashes == ["1", "2", "3"]
    assert len(result) == 3
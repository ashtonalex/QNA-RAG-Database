import os
from dotenv import load_dotenv
import pytest
from unittest.mock import patch
from backend.app.services.rag_service import RAGService

load_dotenv()

@pytest.mark.asyncio
async def test_enhance_query_rewrites(monkeypatch):
    service = RAGService()
    mock_response = {
        "choices": [
            {"message": {"content": "expanded query with synonyms and corrections"}}
        ]
    }

    class MockResponse:
        status = 200

        async def json(self):
            return mock_response

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def post(self, *args, **kwargs):
            return MockResponse()

    with patch("aiohttp.ClientSession", return_value=MockSession()):
        rewritten = await service.enhance_query("orig querry")
        assert rewritten == "expanded query with synonyms and corrections"


@pytest.mark.asyncio
async def test_enhance_query_fallback(monkeypatch):
    service = RAGService()

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def post(self, *args, **kwargs):
            raise Exception("API error")

    with patch("aiohttp.ClientSession", return_value=MockSession()):
        rewritten = await service.enhance_query("orig querry")
        assert rewritten == "orig querry"


@pytest.mark.asyncio
async def test_retrieve_candidates_dedup_and_filter():
    service = RAGService()
    # Mock vector_service.similarity_search
    mock_results = [
        {
            "text": "A",
            "metadata": {"hash": "1", "section_title": "Intro"},
            "score": 0.1,
        },
        {"text": "B", "metadata": {"hash": "2", "section_title": "Body"}, "score": 0.2},
        {
            "text": "A-dup",
            "metadata": {"hash": "1", "section_title": "Intro"},
            "score": 0.3,
        },
    ]

    class DummyVectorService:
        async def similarity_search(self, **kwargs):
            # Simulate metadata filter
            if kwargs.get("metadata_filter") == {"section_title": "Body"}:
                return [mock_results[1]]
            return mock_results

    # No filter: deduplication
    deduped = await service.retrieve_candidates(
        query="q", collection_name="c", vector_service=DummyVectorService()
    )
    assert len(deduped) == 2
    assert deduped[0]["metadata"]["hash"] == "1"
    assert deduped[1]["metadata"]["hash"] == "2"
    # With filter
    filtered = await service.retrieve_candidates(
        query="q",
        collection_name="c",
        metadata_filter={"section_title": "Body"},
        vector_service=DummyVectorService(),
    )
    assert len(filtered) == 1
    assert filtered[0]["metadata"]["hash"] == "2"

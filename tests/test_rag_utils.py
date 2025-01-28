import pytest
from ..rag_utils import retrieve_relevant_chunks, semantic_search
from unittest.mock import patch

@pytest.fixture
def sample_chunks():
    return [
        "AWS S3 bucket documentation",
        "Azure VM documentation",
        "GCP Cloud Storage documentation"
    ]

@pytest.mark.asyncio
async def test_retrieve_relevant_chunks():
    chunks = sample_chunks()
    result = await retrieve_relevant_chunks("Create S3 bucket", chunks)
    assert any("S3" in chunk for chunk in result)

@pytest.mark.asyncio
async def test_semantic_search():
    with patch('chromadb.PersistentClient') as mock_client:
        mock_client.return_value.query.return_value = {
            'documents': [["AWS S3 bucket documentation"]]
        }
        result = await semantic_search("Create S3 bucket")
        assert len(result) > 0
        assert "S3" in result[0]

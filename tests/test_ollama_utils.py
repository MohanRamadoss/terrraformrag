import pytest
from unittest.mock import patch, MagicMock
from ..ollama_utils import generate_code_ollama, prepare_model_prompt, clean_code_response

@pytest.fixture
def mock_ollama_response():
    return {
        "message": {
            "content": "```hcl\nprovider \"aws\" {\n  region = var.region\n}\n```"
        }
    }

@pytest.mark.asyncio
async def test_generate_code_ollama():
    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = mock_ollama_response()
        result = await generate_code_ollama(
            "Create S3 bucket",
            ["AWS S3 bucket documentation"],
            "granite-code:20b"
        )
        assert "provider" in result
        assert "aws" in result

def test_prepare_model_prompt():
    prompt = prepare_model_prompt(
        "Create S3 bucket",
        ["AWS S3 context"],
        "granite-code:20b"
    )
    assert "terraform" in prompt.lower()
    assert "s3 bucket" in prompt.lower()

def test_clean_code_response():
    code = clean_code_response("```hcl\nprovider aws {}\n```")
    assert "provider aws" in code
    assert "```" not in code

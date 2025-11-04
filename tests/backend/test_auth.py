import pytest
from fastapi import HTTPException


from chroma_knowledge_search.backend.app.auth import require_api_key


class TestAuth:
    """Test authentication functionality."""

    @pytest.mark.asyncio
    async def test_valid_api_key(self, mock_secrets):
        """Test authentication with valid API key."""
        result = await require_api_key("test-api-key")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hash length

    @pytest.mark.asyncio
    async def test_missing_api_key(self, mock_secrets):
        """Test authentication with missing API key."""
        with pytest.raises(HTTPException) as exc_info:
            await require_api_key(None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_api_key(self, mock_secrets):
        """Test authentication with invalid API key."""
        with pytest.raises(HTTPException) as exc_info:
            await require_api_key("invalid-key")
        assert exc_info.value.status_code == 403

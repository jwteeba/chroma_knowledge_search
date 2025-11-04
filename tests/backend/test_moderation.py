from unittest.mock import Mock, patch

from chroma_knowledge_search.backend.app.moderation import is_flagged


class TestModeration:
    """Test content moderation."""

    def test_is_flagged_safe_content(self):
        """Test moderation with safe content."""
        with (
            patch("streamlit.secrets") as mock_secrets,
            patch(
                "chroma_knowledge_search.backend.app.moderation.client"
            ) as mock_client,
        ):
            mock_secrets.openai.api_key = "test-key"
            mock_secrets.openai.moderation_model = "text-moderation-latest"

            mock_response = Mock()
            mock_response.results = [Mock(flagged=False)]
            mock_client.moderations.create.return_value = mock_response

            result = is_flagged("This is safe content")

            assert result is False
            mock_client.moderations.create.assert_called_once()

    def test_is_flagged_unsafe_content(self):
        """Test moderation with unsafe content."""
        with (
            patch("streamlit.secrets") as mock_secrets,
            patch(
                "chroma_knowledge_search.backend.app.moderation.client"
            ) as mock_client,
        ):
            mock_secrets.openai.api_key = "test-key"
            mock_secrets.openai.moderation_model = "text-moderation-latest"

            mock_response = Mock()
            mock_response.results = [Mock(flagged=True)]
            mock_client.moderations.create.return_value = mock_response

            result = is_flagged("Unsafe content")

            assert result is True

    def test_is_flagged_no_results(self):
        """Test moderation with no results."""
        with (
            patch("streamlit.secrets") as mock_secrets,
            patch(
                "chroma_knowledge_search.backend.app.moderation.client"
            ) as mock_client,
        ):
            mock_secrets.openai.api_key = "test-key"
            mock_secrets.openai.moderation_model = "text-moderation-latest"

            mock_response = Mock()
            mock_response.results = []
            mock_client.moderations.create.return_value = mock_response

            result = is_flagged("Test content")

            assert result is False

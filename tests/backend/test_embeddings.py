from unittest.mock import Mock, patch

from chroma_knowledge_search.backend.app.embeddings import get_embeddings


class TestEmbeddings:
    """Test embedding generation."""

    def test_get_embeddings_success(self):
        """Test successful embedding generation."""
        with patch(
            "chroma_knowledge_search.backend.app.embeddings.client"
        ) as mock_client:
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1] * 1536),
                Mock(embedding=[0.2] * 1536),
            ]
            mock_client.embeddings.create.return_value = mock_response

            texts = ["Hello world", "Test text"]
            embeddings = get_embeddings(texts)

            assert len(embeddings) == 2
            assert all(len(emb) == 1536 for emb in embeddings)
            mock_client.embeddings.create.assert_called_once()

    def test_get_embeddings_retry(self):
        """Test embedding generation with retry logic."""
        with patch(
            "chroma_knowledge_search.backend.app.embeddings.client"
        ) as mock_client:
            # First call fails, second succeeds
            mock_client.embeddings.create.side_effect = [
                Exception("API Error"),
                Mock(data=[Mock(embedding=[0.1] * 1536)]),
            ]

            embeddings = get_embeddings(["test"])

            assert len(embeddings) == 1
            assert mock_client.embeddings.create.call_count == 2

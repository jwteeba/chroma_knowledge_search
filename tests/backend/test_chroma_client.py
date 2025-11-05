from unittest.mock import Mock, patch

from chroma_knowledge_search.backend.app.chroma_client import (
    get_or_create_collection,
    query,
    upsert_chunks,
)


class TestChromaClient:
    """Test ChromaDB client functionality."""

    def test_get_or_create_collection_exists(self, mock_chroma):
        """Test getting existing collection."""
        collection = get_or_create_collection()

        assert collection is not None
        mock_chroma.get_collection.assert_called_once()

    def test_get_or_create_collection_create_new(self):
        """Test creating new collection when it doesn't exist."""
        with patch(
            "chroma_knowledge_search.backend.app.chroma_client.get_client"
        ) as mock_get_client:
            mock_client = Mock()
            mock_client.get_collection.side_effect = Exception("Not found")
            mock_collection = Mock()
            mock_collection.name = "test-collection"
            mock_client.create_collection.return_value = mock_collection
            mock_get_client.return_value = mock_client

            collection = get_or_create_collection()

            assert collection == mock_collection
            mock_client.create_collection.assert_called_once()

    def test_upsert_chunks(self, mock_chroma):
        """Test upserting document chunks."""
        chunks = [
            {"text": "First chunk", "embedding": [0.1] * 1536},
            {"text": "Second chunk", "embedding": [0.2] * 1536},
        ]

        upsert_chunks("doc-123", chunks, "owner-key")

        mock_collection = mock_chroma.get_collection.return_value
        mock_collection.add.assert_called_once()

        call_args = mock_collection.add.call_args
        assert len(call_args.kwargs["ids"]) == 2
        assert len(call_args.kwargs["embeddings"]) == 2
        assert len(call_args.kwargs["documents"]) == 2
        assert len(call_args.kwargs["metadatas"]) == 2

    def test_query_with_owner_key(self, mock_chroma):
        """Test querying with owner key filter."""
        query_embedding = [0.1] * 1536

        result = query(query_embedding, top_k=3, owner_key="owner-123")

        mock_collection = mock_chroma.get_collection.return_value
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"owner_key": "owner-123"},
        )

    def test_query_without_owner_key(self, mock_chroma):
        """Test querying without owner key filter."""
        query_embedding = [0.1] * 1536

        result = query(query_embedding, top_k=5, owner_key=None)

        mock_collection = mock_chroma.get_collection.return_value
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding], n_results=5, where=None
        )

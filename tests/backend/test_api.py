from unittest.mock import patch, Mock


class TestUploadEndpoint:
    """Test upload API endpoint."""

    def test_upload_missing_api_key(self, client):
        """Test upload without API key."""
        files = {"file": ("test.txt", b"Test content", "text/plain")}

        response = client.post("/api/upload", files=files)

        assert response.status_code == 401

    def test_upload_invalid_api_key(self, client):
        """Test upload with invalid API key."""
        files = {"file": ("test.txt", b"Test content", "text/plain")}
        headers = {"x-api-key": "invalid-key"}

        response = client.post("/api/upload", files=files, headers=headers)

        assert response.status_code == 403

    def test_upload_file_too_large(self, client):
        """Test upload with oversized file."""
        large_content = b"x" * (16 * 1024 * 1024)  # 16MB
        files = {"file": ("large.txt", large_content, "text/plain")}
        headers = {"x-api-key": "test-api-key"}

        response = client.post("/api/upload", files=files, headers=headers)

        assert response.status_code == 413

    def test_upload_empty_file(self, client, mock_openai, mock_chroma):
        """Test upload with empty file."""
        files = {"file": ("empty.txt", b"", "text/plain")}
        headers = {"x-api-key": "test-api-key"}

        response = client.post("/api/upload", files=files, headers=headers)

        assert response.status_code == 400


class TestQueryEndpoint:
    """Test query API endpoint."""

    def test_query_success(self, client, mock_openai, mock_chroma):
        """Test successful query."""
        with patch("chroma_knowledge_search.backend.app.db.get_db") as mock_db, patch(
            "chroma_knowledge_search.backend.app.rag.generate_answer",
            return_value="Test answer",
        ) as mock_rag:
            mock_session = Mock()
            mock_db.return_value.__aenter__.return_value = mock_session

            headers = {"x-api-key": "test-api-key"}
            data = {"query": "What is the content?", "top_k": 5}

            response = client.post("/api/query", json=data, headers=headers)

            assert response.status_code == 200
            result = response.json()
            assert "answer" in result
            assert "sources" in result

    def test_query_missing_api_key(self, client):
        """Test query without API key."""
        data = {"query": "test query"}

        response = client.post("/api/query", json=data)

        assert response.status_code == 401

    def test_query_no_results(self, client, mock_openai):
        """Test query with no matching documents."""
        with patch("chroma_knowledge_search.backend.app.db.get_db") as mock_db, patch(
            "chroma_knowledge_search.backend.app.chroma_client.client"
        ) as mock_client:
            mock_session = Mock()
            mock_db.return_value.__aenter__.return_value = mock_session

            mock_collection = Mock()
            mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
            mock_client.get_collection.return_value = mock_collection

            headers = {"x-api-key": "test-api-key"}
            data = {"query": "nonexistent content"}

            response = client.post("/api/query", json=data, headers=headers)

            assert response.status_code == 200
            result = response.json()
            assert "couldn't find relevant context" in result["answer"]
            assert result["sources"] == []


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

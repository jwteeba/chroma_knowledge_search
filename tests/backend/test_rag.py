from unittest.mock import Mock, patch

from chroma_knowledge_search.backend.app.rag import (
    build_prompt,
    generate_answer,
)


class TestRAG:
    """Test RAG functionality."""

    def test_build_prompt(self):
        """Test prompt building for RAG."""
        chunks = ["First chunk content", "Second chunk content"]
        question = "What is the content?"

        messages = build_prompt(chunks, question)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "First chunk content" in messages[1]["content"]
        assert "What is the content?" in messages[1]["content"]

    def test_generate_answer_success(self):
        """Test successful answer generation."""
        with (
            patch("streamlit.secrets") as mock_secrets,
            patch(
                "chroma_knowledge_search.backend.app.rag.is_flagged",
                return_value=False,
            ),
            patch(
                "chroma_knowledge_search.backend.app.rag.client"
            ) as mock_client,
        ):
            mock_secrets.openai.api_key = "test-key"
            mock_secrets.openai.chat_model = "gpt-3.5-turbo"

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Test answer"))]
            mock_client.chat.completions.create.return_value = mock_response

            chunks = ["Test content"]
            question = "What is this?"

            answer = generate_answer(chunks, question)

            assert answer == "Test answer"
            mock_client.chat.completions.create.assert_called_once()

    def test_generate_answer_flagged_question(self):
        """Test answer generation with flagged question."""
        with (
            patch("streamlit.secrets") as mock_secrets,
            patch(
                "chroma_knowledge_search.backend.app.rag.is_flagged",
                return_value=True,
            ),
            patch(
                "chroma_knowledge_search.backend.app.rag.client"
            ) as mock_client,
        ):
            mock_secrets.openai.api_key = "test-key"

            chunks = ["Test content"]
            question = "Inappropriate question"

            answer = generate_answer(chunks, question)

            assert "can't assist" in answer
            mock_client.chat.completions.create.assert_not_called()

    def test_generate_answer_flagged_response(self):
        """Test answer generation with flagged response."""
        with (
            patch("streamlit.secrets") as mock_secrets,
            patch(
                "chroma_knowledge_search.backend.app.rag.is_flagged",
                side_effect=[False, True],
            ),
            patch(
                "chroma_knowledge_search.backend.app.rag.client"
            ) as mock_client,
        ):
            mock_secrets.openai.api_key = "test-key"
            mock_secrets.openai.chat_model = "gpt-3.5-turbo"

            mock_response = Mock()
            mock_response.choices = [
                Mock(message=Mock(content="Flagged content"))
            ]
            mock_client.chat.completions.create.return_value = mock_response

            chunks = ["Test content"]
            question = "What is this?"

            answer = generate_answer(chunks, question)

            assert "can't share" in answer

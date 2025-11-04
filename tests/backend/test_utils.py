import pytest
import html
from unittest.mock import patch, Mock


from chroma_knowledge_search.backend.app.utils import extract_text_from_file, chunk_text


class TestTextExtraction:
    """Test text extraction utilities."""

    @pytest.mark.asyncio
    async def test_extract_text_from_txt(self):
        """Test text extraction from plain text file."""
        content = b"Hello world"
        result = await extract_text_from_file(content, "test.txt")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_extract_text_from_pdf(self, sample_pdf):
        """Test text extraction from PDF file."""
        result = await extract_text_from_file(sample_pdf, "test.pdf")
        assert (
            "Test content"
            in html.unescape(result.strip('"'))
            .replace("\n", "")
            .replace("\x0c", " ")
            .strip()
        )

    @pytest.mark.asyncio
    async def test_extract_text_from_docx(self):
        """Test text extraction from DOCX file."""
        with patch(
            "chroma_knowledge_search.backend.app.utils.docx.Document"
        ) as mock_doc:
            mock_paragraph = Mock()
            mock_paragraph.text = "DOCX content"
            mock_doc.return_value.paragraphs = [mock_paragraph]

            result = await extract_text_from_file(b"fake docx", "test.docx")
            assert result == "DOCX content"


class TestTextChunking:
    """Test text chunking functionality."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "word1 word2 word3 word4 word5"
        chunks = chunk_text(text, chunk_size=3, overlap=1)

        assert len(chunks) > 1
        assert all("text" in chunk for chunk in chunks)
        assert chunks[0]["text"] == "word1 word2 word3"

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=3, overlap=1)
        assert chunks == []

    def test_chunk_text_single_word(self):
        """Test chunking single word."""
        chunks = chunk_text("word", chunk_size=3, overlap=1)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "word"

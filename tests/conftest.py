import pytest
import asyncio
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


from chroma_knowledge_search.backend.app.models import Base
from chroma_knowledge_search.backend.app.db import get_db
from chroma_knowledge_search.backend.app.main import app


@pytest.fixture
async def test_db():
    """Create test database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    AsyncSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async def get_test_db():
        async with AsyncSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = get_test_db
    yield engine
    app.dependency_overrides.clear()


@pytest.fixture
def mock_secrets():
    """Mock Streamlit secrets."""
    with patch("streamlit.secrets") as mock:
        mock.fastapi.api_key = "test-api-key"
        mock.openai.api_key = "test-openai-key"
        mock.openai.embed_model = "text-embedding-ada-002"
        mock.openai.chat_model = "gpt-3.5-turbo"
        mock.openai.moderation_model = "text-moderation-latest"
        mock.chromadb.chroma_collection = "test-collection"
        mock.sqlite.db_url = "sqlite+aiosqlite:///:memory:"
        mock.cors.allow_origins = ["*"]
        yield mock


@pytest.fixture
def client(mock_secrets):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_pdf():
    """Create sample PDF content."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test content) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000206 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n299\n%%EOF"


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch("chroma_knowledge_search.backend.app.embeddings.client") as mock_client:
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        mock_chat_response = Mock()
        mock_chat_response.choices = [Mock(message=Mock(content="Test answer"))]
        mock_client.chat.completions.create.return_value = mock_chat_response

        mock_mod_response = Mock()
        mock_mod_response.results = [Mock(flagged=False)]
        mock_client.moderations.create.return_value = mock_mod_response

        yield mock_client


@pytest.fixture
def mock_chroma():
    """Mock ChromaDB client."""
    with patch(
        "chroma_knowledge_search.backend.app.chroma_client.client"
    ) as mock_client:
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Test document content"]],
            "metadatas": [[{"document_id": "test-doc-id"}]],
        }
        mock_client.get_collection.return_value = mock_collection
        mock_client.create_collection.return_value = mock_collection
        yield mock_client

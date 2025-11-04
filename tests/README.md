# Test Suite for Chroma Knowledge Search

This directory contains comprehensive tests for the Chroma Knowledge Search package.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── backend/                 # Backend API tests
│   ├── test_api.py         # API endpoint tests
│   ├── test_auth.py        # Authentication tests
│   ├── test_utils.py       # Utility function tests
│   ├── test_embeddings.py  # Embedding generation tests
│   ├── test_rag.py         # RAG functionality tests
│   ├── test_moderation.py  # Content moderation tests
│   └── test_chroma_client.py # ChromaDB client tests
├── frontend/               # Frontend UI tests
│   └── test_streamlit_ui.py # Selenium UI tests
├── test_integration.py     # Integration tests
├── test_models.py          # Database model tests
└── test_schemas.py         # Pydantic schema tests
```

## Running Tests

### Prerequisites

1. Install dependencies:
```bash
pip install -e .
```

2. For UI tests, install ChromeDriver:
```bash
# Ubuntu/Debian
sudo apt-get install chromium-chromedriver

# macOS
brew install chromedriver

# Or download from: https://chromedriver.chromium.org/
```

### Test Commands

#### Run all tests:
```bash
python run_tests.py
# or
make test
```

#### Run specific test categories:
```bash
# Unit tests only
make test-unit

# Integration tests
make test-integration

# UI tests (requires ChromeDriver)
make test-ui

# With coverage report
make test-coverage
```

#### Run individual test files:
```bash
pytest tests/backend/test_api.py -v
pytest tests/frontend/test_streamlit_ui.py -v
```

## Test Categories

### Unit Tests
- **API Tests**: Test FastAPI endpoints with mocked dependencies
- **Auth Tests**: Test API key authentication and authorization
- **Utils Tests**: Test text extraction and chunking utilities
- **Embeddings Tests**: Test OpenAI embedding generation with retry logic
- **RAG Tests**: Test retrieval-augmented generation functionality
- **Moderation Tests**: Test content safety checks
- **ChromaDB Tests**: Test vector database operations
- **Model Tests**: Test SQLAlchemy database models
- **Schema Tests**: Test Pydantic request/response schemas

### Integration Tests
- **End-to-End Flow**: Test complete upload → query workflow
- **Server Integration**: Test with actual FastAPI server instance

### UI Tests (Selenium)
- **Page Loading**: Test Streamlit app loads correctly
- **Form Interaction**: Test user input and form submission
- **API Key Validation**: Test UI behavior with/without API key
- **Upload Interface**: Test file upload functionality
- **Query Interface**: Test question input and results display

## Test Configuration

### Fixtures
- `mock_secrets`: Mocks Streamlit secrets for testing
- `test_db`: In-memory SQLite database for testing
- `client`: FastAPI test client
- `mock_openai`: Mocked OpenAI API responses
- `mock_chroma`: Mocked ChromaDB operations
- `driver`: Selenium WebDriver for UI tests

### Markers
- `unit`: Unit tests (default)
- `integration`: Integration tests
- `selenium`: UI tests requiring ChromeDriver
- `slow`: Long-running tests

## Coverage

The test suite aims for high coverage of:
- ✅ API endpoints and error handling
- ✅ Authentication and authorization
- ✅ File processing and text extraction
- ✅ Embedding generation and vector operations
- ✅ RAG pipeline and answer generation
- ✅ Content moderation and safety
- ✅ Database operations
- ✅ UI components and user interactions

## Mocking Strategy

External dependencies are mocked to ensure:
- Fast test execution
- Reliable test results
- No external API costs
- Isolated component testing

Mocked services:
- OpenAI API (embeddings, chat, moderation)
- ChromaDB operations
- Streamlit secrets
- File system operations

## Continuous Integration

Tests are designed to run in CI environments with:
- Headless browser support for UI tests
- In-memory databases
- Mocked external services
- Comprehensive error reporting
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from chroma_knowledge_search.backend.app.auth import require_api_key
from chroma_knowledge_search.backend.app.chroma_client import (
    query as chroma_query,
)
from chroma_knowledge_search.backend.app.chroma_client import upsert_chunks
from chroma_knowledge_search.backend.app.db import get_db
from chroma_knowledge_search.backend.app.embeddings import get_embeddings
from chroma_knowledge_search.backend.app.models import Document
from chroma_knowledge_search.backend.app.rag import generate_answer
from chroma_knowledge_search.backend.app.schemas import (
    QueryRequest,
    QueryResult,
    UploadResponse,
)
from chroma_knowledge_search.backend.app.utils import (
    chunk_text,
    extract_text_from_file,
)

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    owner_key: str = Depends(require_api_key),
):
    """Upload and process a document for knowledge search.

    Extracts text from uploaded file, chunks it, generates embeddings,
    stores in vector database and saves metadata.

    Args:
        file (UploadFile): File to upload and process
        db (AsyncSession): Database session
        owner_key (str): API key for authentication

    Returns:
        UploadResponse: Document ID and number of chunks indexed

    Raises:
        HTTPException: If file too large, no text found, or processing fails
    """

    # Size validation
    MAX_FILE_SIZE_MB = 15
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    # Extract text
    text = await extract_text_from_file(content, file.filename)
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found")

    # Chunking
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = [c for c in chunks if c["text"].strip()]  # ✅ remove empty chunks
    if not chunks:
        raise HTTPException(
            status_code=400, detail="No valid text chunks extracted"
        )

    # Embeddings
    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts)

    if len(embeddings) != len(chunks):
        raise HTTPException(
            status_code=500,
            detail=f"Embedding mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings",
        )

    # Attach embeddings
    for c, emb in zip(chunks, embeddings):
        c["embedding"] = emb

    # Store vectors in Chroma
    document_id = str(uuid.uuid4())
    upsert_chunks(document_id, chunks, owner_key=owner_key)

    # Store doc metadata
    doc = Document(
        id=document_id,
        owner_key=owner_key,
        filename=file.filename,
        text_preview=text[:1000],
    )
    db.add(doc)
    await db.commit()

    return UploadResponse(document_id=document_id, chunks_indexed=len(chunks))


@router.post("/query", response_model=QueryResult)
async def query_docs(
    req: QueryRequest,
    db: AsyncSession = Depends(get_db),
    owner_key: str = Depends(require_api_key),
):
    """Query documents using semantic search and generate AI answer.

    Converts query to embedding, searches vector database for relevant chunks,
    and generates contextual answer using retrieved documents.

    Args:
        req (QueryRequest): Query request with text and optional top_k
        db (AsyncSession): Database session
        owner_key (str): API key for authentication

    Returns:
        QueryResult: Generated answer and source document IDs
    """

    qemb = get_embeddings([req.query])[0]
    res = chroma_query(qemb, top_k=req.top_k, owner_key=owner_key)

    docs = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]

    if not docs:
        return QueryResult(
            answer="I couldn't find relevant context for your question.",
            sources=[],
        )

    answer = generate_answer(docs, req.query)
    sources = [
        m.get("document_id")
        for m in metadatas
        if isinstance(m, dict) and "document_id" in m
    ]

    # Deduplicate while preserving order
    sources = list(dict.fromkeys(sources))

    return QueryResult(answer=answer, sources=sources)

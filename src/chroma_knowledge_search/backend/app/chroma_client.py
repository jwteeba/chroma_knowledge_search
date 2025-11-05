import os

import chromadb
import streamlit as st

from chroma_knowledge_search.backend.app.logging_config import get_logger

logger = get_logger(__name__)
_client = None


def get_client():
    """Get ChromaDB client instance."""
    global _client
    if _client is None:
        # Use local client in test environment
        if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in os.environ.get(
            "_", ""
        ):
            logger.info("Using local ChromaDB client for testing")
            _client = chromadb.Client()
        else:
            # Use Chroma Cloud if credentials are available
            try:
                if (
                    hasattr(st.secrets, "chromadb")
                    and hasattr(st.secrets.chromadb, "chroma_api_key")
                    and st.secrets.chromadb.chroma_api_key
                ):
                    logger.info("Using ChromaDB Cloud client")
                    _client = chromadb.CloudClient(
                        api_key=st.secrets.chromadb.chroma_api_key,
                        tenant=st.secrets.chromadb.chroma_tenant,
                        database=st.secrets.chromadb.chroma_database,
                    )
                else:
                    logger.info("Using local ChromaDB client")
                    _client = chromadb.Client()
            except Exception as e:
                logger.warning(
                    f"Failed to connect to ChromaDB Cloud, using local client: {e}"
                )
                _client = chromadb.Client()
    return _client


def get_or_create_collection():
    """Get existing collection or create new one.

    Returns:
        Collection: ChromaDB collection instance
    """
    client = get_client()
    try:
        collection = client.get_collection(
            st.secrets.chromadb.chroma_collection
        )
        logger.debug(
            f"Retrieved existing collection: {st.secrets.chromadb.chroma_collection}"
        )
        return collection
    except Exception as e:
        logger.info(
            f"Creating new collection: {st.secrets.chromadb.chroma_collection}"
        )
        return client.create_collection(
            st.secrets.chromadb.chroma_collection,
            metadata={"description": "Knowledge search collection"},
        )


def upsert_chunks(document_id: str, chunks: list[dict], owner_key: str):
    """Store document chunks with embeddings in vector database.

    Args:
        document_id (str): Unique document identifier
        chunks (list[dict]): Text chunks with embeddings
        owner_key (str): Owner key for access control
    """
    logger.info(f"Upserting {len(chunks)} chunks for document {document_id}")
    col = get_or_create_collection()
    ids = [f"{document_id}-{i}" for i, _ in enumerate(chunks)]
    metadatas = [
        {"document_id": document_id, "owner_key": owner_key} for _ in chunks
    ]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["text"] for c in chunks]
    col.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    logger.debug(f"Successfully stored {len(chunks)} chunks")


def query(query_embedding, top_k=5, owner_key: str | None = None):
    """Search for similar chunks using vector similarity.

    Args:
        query_embedding: Query vector embedding
        top_k (int): Number of results to return
        owner_key (str, optional): Filter by owner key

    Returns:
        dict: Query results with documents and metadata
    """
    logger.debug(
        f"Querying ChromaDB with top_k={top_k}, owner_key={'set' if owner_key else 'none'}"
    )
    col = get_or_create_collection()
    where = {"owner_key": owner_key} if owner_key else None
    results = col.query(
        query_embeddings=[query_embedding], n_results=top_k, where=where
    )
    logger.debug(
        f"Query returned {len(results.get('documents', [[]])[0])} results"
    )
    return results

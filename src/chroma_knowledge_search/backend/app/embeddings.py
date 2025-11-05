import streamlit as st
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from chroma_knowledge_search.backend.app.logging_config import get_logger

logger = get_logger(__name__)
client = OpenAI(api_key=st.secrets.openai.api_key)


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for text using OpenAI API with retry logic.

    Args:
        texts (list[str]): List of texts to embed

    Returns:
        list[list[float]]: List of embedding vectors
    """
    logger.debug(f"Generating embeddings for {len(texts)} texts")
    resp = client.embeddings.create(
        model=st.secrets.openai.embed_model, input=texts
    )
    logger.debug(f"Generated {len(resp.data)} embeddings")
    return [d.embedding for d in resp.data]

import os
from openai import OpenAI

from chroma_knowledge_search.backend.app.logging_config import get_logger
from chroma_knowledge_search.backend.app.moderation import is_flagged
from chroma_knowledge_search.backend.app.config import load_config

logger = get_logger(__name__)

load_config()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_chat_model = os.getenv("OPENAI_CHAT_MODEL")
client = OpenAI(api_key=openai_api_key)

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Use ONLY the provided context to answer. "
    "If the answer is not in the context, say you don't know."
)


def build_prompt(context_chunks: list[str], question: str) -> list[dict]:
    """Build chat messages for RAG prompt.

    Args:
        context_chunks (list[str]): Retrieved context chunks
        question (str): User question

    Returns:
        list[dict]: Chat messages for OpenAI API
    """
    context = "\n\n".join(
        [f"[Chunk {i + 1}] {c}" for i, c in enumerate(context_chunks)]
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer using only the context. Cite chunk numbers in brackets when relevant."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(context_chunks: list[str], question: str) -> str:
    """Generate answer using retrieved context and safety checks.

    Args:
        context_chunks (list[str]): Retrieved document chunks
        question (str): User question

    Returns:
        str: Generated answer or safety message
    """
    logger.debug(
        f"Generating answer for question with {len(context_chunks)} context chunks"
    )

    # Safety pre-check on user question
    if is_flagged(question):
        logger.warning("Question flagged by moderation")
        return "I'm sorry, I can't assist with that request."

    messages = build_prompt(context_chunks, question)
    resp = client.chat.completions.create(
        model=openai_chat_model,
        messages=messages,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content

    # Safety post-check on model answer
    if is_flagged(answer):
        logger.warning("Generated answer flagged by moderation")
        return "I'm sorry, I can't share that content."

    logger.debug("Answer generated successfully")
    return answer

import os
from openai import OpenAI
from chroma_knowledge_search.backend.app.config import load_config

load_config()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)


def is_flagged(text: str) -> bool:
    """Check if text violates OpenAI's usage policies.

    Args:
        text (str): Text to moderate

    Returns:
        bool: True if content is flagged as unsafe, False otherwise
    """

    mod = client.moderations.create(model="omni-moderation-latest", input=text)
    result = getattr(mod, "results", None)
    if result and len(result) > 0:
        flagged = getattr(result[0], "flagged", False)
        return bool(flagged)
    return False

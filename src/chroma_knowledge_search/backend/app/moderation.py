from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets.openai.api_key)


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

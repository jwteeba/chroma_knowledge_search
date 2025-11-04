import hashlib

import streamlit as st
from fastapi import Header, HTTPException, status


async def require_api_key(x_api_key: str = Header(None)) -> str:
    """Validate API key and return owner key hash.

    Args:
        x_api_key (str): API key from request header

    Returns:
        str: SHA256 hash of the API key for owner isolation

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key"
        )
    if x_api_key != st.secrets.fastapi.api_key:
        # Optionally use constant-time compare
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key"
        )
    # Per-key isolation tag
    return hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()

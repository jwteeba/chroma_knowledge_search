import io
from typing import List
from pdfminer.high_level import extract_text
import docx

SUPPORTED_EXTS = (".pdf", ".txt", ".docx")


async def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract text from uploaded file based on extension.

    Args:
        file_bytes (bytes): File content as bytes
        filename (str): Original filename with extension

    Returns:
        str: Extracted text content
    """
    fname = filename.lower()
    if fname.endswith(".pdf"):
        with io.BytesIO(file_bytes) as f:
            return extract_text(f)
    if fname.endswith(".docx"):
        with io.BytesIO(file_bytes) as f:
            doc = docx.Document(f)
            return "\n".join(p.text for p in doc.paragraphs)
    return file_bytes.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[dict]:
    """Split text into overlapping chunks.

    Args:
        text (str): Input text to chunk
        chunk_size (int): Maximum words per chunk
        overlap (int): Number of overlapping words

    Returns:
        List[dict]: List of text chunks with 'text' key
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append({"text": chunk})
        i += step
    return chunks

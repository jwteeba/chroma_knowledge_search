from typing import List

from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    chunks_indexed: int


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResult(BaseModel):
    answer: str
    sources: List[str]

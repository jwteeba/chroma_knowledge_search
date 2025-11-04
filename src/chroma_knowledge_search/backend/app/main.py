from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chroma_knowledge_search.backend.app.api import router as api_router
import streamlit as st
from chroma_knowledge_search.backend.app.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Chroma Knowledge Search (API Key)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=st.secrets.cors.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health():
    """Health check endpoint.

    Returns:
        dict: Status response
    """
    return {"status": "ok"}

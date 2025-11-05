from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chroma_knowledge_search.backend.app.config import (
    load_config,
    get_allow_origins,
)

from chroma_knowledge_search.backend.app.api import router as api_router
from chroma_knowledge_search.backend.app.db import init_db
from chroma_knowledge_search.backend.app.logging_config import (
    get_logger,
    setup_logging,
)

setup_logging()
logger = get_logger(__name__)

load_config()
allow_origins = get_allow_origins()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down application")


app = FastAPI(title="Chroma Knowledge Search (API Key)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
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
    logger.debug("Health check requested")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    import os

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

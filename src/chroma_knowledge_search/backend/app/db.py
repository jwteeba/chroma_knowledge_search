import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from chroma_knowledge_search.backend.app.models import Base

_engine = None
_session_local = None


def get_engine():
    """Get or create database engine."""
    global _engine, _session_local
    if _engine is None:
        try:
            from chroma_knowledge_search.backend.app.config import load_config

            load_config()
        except Exception:
            pass  # Config loading may fail in tests

        db_url = os.getenv("DB_URL", "sqlite+aiosqlite:///:memory:")
        _engine = create_async_engine(db_url, echo=False, future=True)
        _session_local = sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
    return _engine, _session_local


async def init_db():
    """Initialize database tables."""
    engine, _ = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Get database session dependency.

    Yields:
        AsyncSession: Database session
    """
    _, session_local = get_engine()
    async with session_local() as session:
        yield session

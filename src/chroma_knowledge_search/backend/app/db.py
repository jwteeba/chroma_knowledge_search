import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from chroma_knowledge_search.backend.app.models import Base
from chroma_knowledge_search.backend.app.config import load_config

# Load configuration
load_config()
db_url = os.getenv("DB_URL")

engine = create_async_engine(db_url, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Get database session dependency.

    Yields:
        AsyncSession: Database session
    """
    async with AsyncSessionLocal() as session:
        yield session

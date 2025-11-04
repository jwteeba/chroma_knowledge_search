from sqlalchemy import Column, String, DateTime, func, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, index=True)
    owner_key = Column(String, index=True, nullable=False)  # hex sha256 of API key
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    text_preview = Column(Text, nullable=True)

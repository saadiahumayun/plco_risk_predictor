# app/db/session.py
"""
Database session configuration.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import logging
import os

from app.core.config import settings

logger = logging.getLogger(__name__)


def _create_sqlite_engine():
    """Create an in-memory SQLite engine."""
    logger.info("Using in-memory SQLite database")
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )


def _create_postgres_engine(database_url: str):
    """Create a PostgreSQL engine with connection test."""
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600,
        echo=False,
    )
    # Test the connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Connected to PostgreSQL database")
    return engine


# Determine database URL
database_url = os.environ.get("DATABASE_URL", settings.DATABASE_URL)

# Check if we should use SQLite (only for test mode)
use_sqlite = (
    settings.ENVIRONMENT == "test" or
    database_url == "postgresql://user:password@localhost/breast_cancer_db"
)

if use_sqlite:
    engine = _create_sqlite_engine()
else:
    try:
        engine = _create_postgres_engine(database_url)
    except Exception as e:
        logger.warning(f"Failed to connect to PostgreSQL: {e}")
        logger.info("Falling back to in-memory SQLite database")
        engine = _create_sqlite_engine()

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """
    Get database session.
    
    This is a dependency that can be used in FastAPI routes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    from app.db.models import Base
    
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.warning(f"Failed to create database tables: {e}")
        logger.info("App will continue without database persistence")
# app/db/session.py
"""
Database session configuration.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create database engine
if settings.ENVIRONMENT == "test" or settings.ENVIRONMENT == "demo":
    # Use in-memory SQLite for tests and demo mode
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    logger.info("Using in-memory SQLite database for demo/test mode")
else:
    # Production database
    try:
        engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=10,        # Number of connections to maintain
            max_overflow=20,     # Maximum overflow connections
            pool_recycle=3600,   # Recycle connections after 1 hour
            echo=False,          # Set to True for SQL query logging
        )
    except Exception as e:
        logger.warning(f"Failed to connect to database: {e}")
        logger.info("Falling back to in-memory SQLite database")
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )

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
    
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
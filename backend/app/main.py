# app/main.py
"""
Main FastAPI application for breast cancer risk prediction API.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from app.core.config import settings
from app.api import routes
from app.core.logging import setup_logging
from app.services.ml_service import ml_service
from app.services.monitoring import monitoring_service

# Optional imports for non-demo environments
try:
    from prometheus_client import make_asgi_app
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    
try:
    from app.db.base import init_db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting up breast cancer risk prediction API...")
    
    # Initialize database (with graceful fallback)
    if DB_AVAILABLE:
        try:
            init_db()
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
            logger.info("Continuing without database - predictions won't be saved")
    else:
        logger.info("Database module not available - predictions won't be saved")
    
    # Load ML models
    logger.info("Loading ML models from MLflow...")
    # Models are loaded automatically when ml_service is imported
    
    # Start monitoring
    if settings.ENABLE_MONITORING:
        monitoring_service.start()
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    if settings.ENABLE_MONITORING:
        monitoring_service.stop()


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan
)

# Set up CORS
cors_origins = settings.get_cors_origins()
if "*" in cors_origins:
    # Allow all origins for demo/production deployment
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # Must be False when allow_origins=["*"]
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add Prometheus metrics endpoint (if available)
if PROMETHEUS_AVAILABLE:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

# Include API routes
app.include_router(routes.api_router, prefix=settings.API_V1_STR)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "details": str(exc) if settings.DEBUG else None
        }
    )

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Incoming {request.method} request to {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"Completed {request.method} {request.url.path} "
        f"with status {response.status_code} in {process_time:.2f}ms"
    )
    
    # Add custom headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Model-Version"] = ml_service.ga_model_version or "unknown"
    
    return response

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "healthy",
        "docs": f"{settings.API_V1_STR}/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy"}


# Database health check endpoint
@app.get("/health/db")
async def database_health_check():
    """Check database connection."""
    from app.db.session import engine
    from sqlalchemy import text
    import os
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        
        # Get database URL (masked)
        db_url = os.environ.get("DATABASE_URL", settings.DATABASE_URL)
        masked_url = db_url[:20] + "..." if len(db_url) > 20 else db_url
        
        return {
            "status": "connected",
            "database": "postgresql" if "postgresql" in db_url else "sqlite",
            "url_preview": masked_url,
            "environment": settings.ENVIRONMENT
        }
    except Exception as e:
        return {
            "status": "disconnected",
            "error": str(e),
            "environment": settings.ENVIRONMENT
        }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
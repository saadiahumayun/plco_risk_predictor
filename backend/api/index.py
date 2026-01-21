"""
Vercel serverless function entry point for FastAPI.
"""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the app module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables for Vercel deployment
import os
os.environ.setdefault("ENVIRONMENT", "demo")
os.environ.setdefault("USE_MODEL_REGISTRY", "false")

# Import the FastAPI app
from app.main import app

# Vercel expects a handler - the app object serves as the ASGI handler
handler = app


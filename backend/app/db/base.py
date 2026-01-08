# app/db/base.py
"""
Database base configuration and initialization.
"""
from app.db.session import init_db
from app.db.models import Base, User, Prediction, ModelRegistry, FeatureDistribution

# Import all models to ensure they're registered with Base
__all__ = [
    "Base",
    "User", 
    "Prediction",
    "ModelRegistry",
    "FeatureDistribution",
    "init_db"
]
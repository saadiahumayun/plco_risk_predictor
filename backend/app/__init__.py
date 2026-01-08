# app/__init__.py
"""
Breast Cancer Risk Prediction API

A production-ready API for predicting 5-year breast cancer risk using
GA-optimized and baseline models trained with MLflow.
"""

__version__ = "1.0.0"
__author__ = "Breast Cancer Risk Prediction Team"
__license__ = "MIT"

# Module imports
from app.core.config import settings
from app.services.ml_service import ml_service
from app.services.mlflow_service import mlflow_service

__all__ = ["settings", "ml_service", "mlflow_service"]
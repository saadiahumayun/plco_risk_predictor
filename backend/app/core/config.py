# app/core/config.py
"""
Core configuration for the breast cancer risk prediction API.
Integrates with MLflow for model serving and tracking.
"""
from typing import List, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import field_validator
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load .env file from backend directory
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Breast Cancer Risk Predictor"
    VERSION: str = "1.0.0"
    
    # CORS Configuration - use string type to avoid parsing issues
    BACKEND_CORS_ORIGINS: str = "*"
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, list):
            return ",".join(v)
        return str(v) if v else "*"
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins as a list."""
        if self.BACKEND_CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS.split(",")]
    
    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/breast_cancer_db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_EXPIRE_SECONDS: int = 3600  # 1 hour
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_S3_ENDPOINT_URL: Optional[str] = None
    MLFLOW_EXPERIMENT_NAME: str = "breast_cancer_production"
    
    # Model Configuration
    GA_MODEL_NAME: str = "F1_MultiPopGA_BreastCancer"
    BASELINE_MODEL_NAME: str = "breast_cancer_baseline_model"
    MODEL_STAGE: str = ""  # Empty string = load latest version. Options: None, Staging, Production, Archived
    
    # Model Registry
    USE_MODEL_REGISTRY: bool = True
    MODEL_REGISTRY_URI: Optional[str] = None  # If None, uses MLFLOW_TRACKING_URI
    
    # Feature Configuration (from your GA results)
    # Stored as comma-separated string to avoid env parsing issues
    _GA_FEATURES_STR: str = "educat,marital,pipe,cigar,sisters,fmenstr,menstrs,miscar,tubal,uterine_fib,lmenstr,prega,thorm,hyperten_f,bronchit_f,diabetes_f,arthrit_f,gallblad_f,bq_age,hyster_f,ovariesr_f,bcontr_f,horm_f,smoked_f,rsmoker_f,cigpd_f,filtered_f,cig_years,bmi_20,bmi_curr,height_f,colon_comorbidity,fh_cancer,entryage_dhq,ph_any_bq,ph_any_dhq,ph_any_sqx,ph_any_trial,entrydays_bq,entrydays_dhq,arm,age"
    
    @property
    def GA_SELECTED_FEATURES(self) -> List[str]:
        return [f.strip() for f in self._GA_FEATURES_STR.split(",")]
    
    @property
    def BASELINE_FEATURES(self) -> List[str]:
        return []  # Will load all available features
    
    # Clinical Thresholds (based on clinical guidelines)
    # Gail model uses 1.67% as threshold for chemoprevention
    HIGH_RISK_THRESHOLD: float = 0.03  # 3% 5-year risk
    MODERATE_RISK_THRESHOLD: float = 0.015  # 1.5% 5-year risk
    
    # Model Performance Thresholds (for monitoring)
    MIN_AUC_THRESHOLD: float = 0.85
    MIN_PRECISION_THRESHOLD: float = 0.70
    MIN_RECALL_THRESHOLD: float = 0.65
    MAX_LATENCY_MS: int = 500
    
    # Monitoring Configuration
    ENABLE_MONITORING: bool = False  # Disabled for demo mode
    PROMETHEUS_PORT: int = 8001
    LOG_PREDICTIONS: bool = False  # Disabled for demo mode
    DRIFT_CHECK_FREQUENCY: int = 1000  # Check drift every N predictions
    
    # Data Validation
    MIN_AGE: int = 20
    MAX_AGE: int = 100
    MIN_BMI: float = 15.0
    MAX_BMI: float = 60.0
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" or "text"
    
    # Environment (set via .env file or environment variable)
    ENVIRONMENT: str = "development"  # demo, development, staging, production
    DEBUG: bool = True
    
    model_config = {
        "case_sensitive": True,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Create settings instance
settings = Settings()

# MLflow-specific environment variables
os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI
if settings.MLFLOW_S3_ENDPOINT_URL:
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT_URL

# Feature mappings for backward compatibility
FEATURE_MAPPINGS = {
    # Map frontend names to dataset names
    "age": "age",
    "bmi": "bmi_curr",
    "age_at_menarche": "fmenstr",
    "age_at_first_birth": "fchilda",
    "race": "race7",
    "family_history": "fh_cancer",
    # Add more mappings as needed
}

# Clinical feature groups
CLINICAL_FEATURES = {
    "demographics": ["age", "race7"],
    "reproductive": ["fmenstr", "fchilda", "livec", "menstrs"],
    "body_metrics": ["bmi_curr", "bmi_20", "weight_f", "height_f"],
    "medical_history": ["ph_any_trial", "ph_any_sqx", "ph_any_dhq", "fh_cancer"],
    "lifestyle": ["ibup"],
    "study_related": ["center", "arm", "rndyear", "reconsent_outcome"]
}
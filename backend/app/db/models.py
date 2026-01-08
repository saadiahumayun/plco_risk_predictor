# app/db/models.py
"""
SQLAlchemy database models.
"""
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, JSON, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255))
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    predictions = relationship("Prediction", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")


class Prediction(Base):
    """Model for storing prediction results."""
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    patient_id = Column(String(100), nullable=True, index=True)  # MR Number
    
    # Risk prediction results
    risk_score = Column(Float, nullable=False)
    risk_category = Column(String(20), nullable=False)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    percentile = Column(Integer)
    relative_risk = Column(Float)
    
    # Model information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), default="ga_optimized")
    
    # Input features
    input_features = Column(JSONB, nullable=False)
    preprocessed_features = Column(JSONB)
    
    # Analysis results
    feature_importance = Column(JSONB)
    top_risk_factors = Column(JSONB)
    comparison_results = Column(JSONB)
    
    # Performance metrics
    processing_time_ms = Column(Float)
    inference_time_ms = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="predictions")


class ModelRegistry(Base):
    """Model registry for tracking deployed models."""
    __tablename__ = "model_registry"
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    model_stage = Column(String(20))
    mlflow_run_id = Column(String(255))
    
    # Model metadata
    algorithm = Column(String(50))
    features_used = Column(JSONB)
    feature_count = Column(Integer)
    training_date = Column(DateTime(timezone=True))
    
    # Performance metrics
    auc = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    
    # Deployment info
    deployed_at = Column(DateTime(timezone=True))
    deployed_by = Column(String(255))
    is_active = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class FeatureDistribution(Base):
    """Feature distribution tracking for drift detection."""
    __tablename__ = "feature_distributions"
    
    id = Column(Integer, primary_key=True)
    feature_name = Column(String(100), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)
    hour = Column(Integer, default=0)
    
    # Distribution statistics
    mean = Column(Float)
    std_dev = Column(Float)
    min_value = Column(Float)
    max_value = Column(Float)
    p25 = Column(Float)
    p50 = Column(Float)
    p75 = Column(Float)
    
    # Counts
    sample_count = Column(Integer)
    missing_count = Column(Integer, default=0)
    
    # Drift metrics
    drift_score = Column(Float)
    is_drifted = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AnalyticsEvent(Base):
    """Analytics events tracking."""
    __tablename__ = "analytics_events"
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False)
    event_data = Column(JSONB)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    prediction_id = Column(UUID(as_uuid=True))
    
    # Event metadata
    ip_address = Column(INET)
    user_agent = Column(Text)
    referrer = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class APIKey(Base):
    """API keys for programmatic access."""
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    key_hash = Column(String(255), unique=True, nullable=False)
    name = Column(String(255))
    scopes = Column(JSONB, default=["predict"])
    
    # Usage tracking
    last_used_at = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    
    # Lifecycle
    expires_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="api_keys")


class PerformanceMetric(Base):
    """Performance monitoring metrics."""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # API metrics
    endpoint = Column(String(255))
    method = Column(String(10))
    status_code = Column(Integer)
    response_time_ms = Column(Float)
    
    # Model metrics
    model_name = Column(String(100))
    model_latency_ms = Column(Float)
    batch_size = Column(Integer, default=1)
    
    # System metrics
    cpu_usage_percent = Column(Float)
    memory_usage_mb = Column(Float)
    
    # Error info
    error_type = Column(String(255))
    error_message = Column(Text)
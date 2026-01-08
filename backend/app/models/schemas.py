# app/models/schemas.py
"""
Pydantic models for request/response validation and data contracts.
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class RiskCategory(str, Enum):
    """Risk category enumeration."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class RecommendationType(str, Enum):
    """Types of clinical recommendations."""
    SCREENING = "screening"
    LIFESTYLE = "lifestyle"
    MEDICAL = "medical"
    GENETIC = "genetic"


# Input schemas
class Demographics(BaseModel):
    """Patient demographics."""
    age: int = Field(..., ge=20, le=100, description="Patient age in years")
    race: str = Field(..., description="Race/ethnicity (white, black, hispanic, asian, other)")
    education: Optional[str] = Field(None, description="Education level")
    marital_status: Optional[str] = Field(None, description="Marital status")
    occupation: Optional[str] = Field(None, description="Occupation")
    
    @validator('race')
    def validate_race(cls, v):
        valid_races = ['white', 'black', 'hispanic', 'asian', 'other']
        if v.lower() not in valid_races:
            raise ValueError(f"Race must be one of: {', '.join(valid_races)}")
        return v.lower()


class ReproductiveHistory(BaseModel):
    """Reproductive history information."""
    age_at_menarche: int = Field(..., ge=8, le=20, description="Age at first menstruation")
    age_at_first_birth: Optional[int] = Field(None, ge=15, le=50, description="Age at first live birth")
    number_of_live_births: int = Field(..., ge=0, le=20, description="Number of live births")
    breastfeeding_months: Optional[int] = Field(None, ge=0, description="Total months of breastfeeding")
    miscarriages: Optional[int] = Field(None, ge=0, le=10, description="Number of miscarriages")
    first_degree_bc: int = Field(..., ge=0, le=10, description="Number of first-degree relatives with breast cancer")
    family_history_details: Optional[List[int]] = Field(None, description="Ages when relatives were diagnosed")


class BodyMetrics(BaseModel):
    """Body measurements and metrics."""
    current_bmi: float = Field(..., ge=15.0, le=60.0, description="Current BMI")
    bmi_at_age_20: Optional[float] = Field(None, ge=15.0, le=60.0, description="BMI at age 20")
    bmi_at_age_50: Optional[float] = Field(None, ge=15.0, le=60.0, description="BMI at age 50")
    height_cm: float = Field(..., ge=120, le=220, description="Height in centimeters")
    weight_kg: float = Field(..., ge=30, le=300, description="Current weight in kilograms")
    
    @validator('current_bmi')
    def validate_bmi(cls, v, values):
        if 'height_cm' in values and 'weight_kg' in values:
            calculated_bmi = values['weight_kg'] / ((values['height_cm'] / 100) ** 2)
            if abs(v - calculated_bmi) > 2:  # Allow some tolerance
                raise ValueError(f"BMI doesn't match height/weight. Calculated: {calculated_bmi:.1f}")
        return v


class MedicalHistory(BaseModel):
    """Medical history information."""
    personal_cancer_history: bool = Field(..., description="Any previous cancer diagnosis")
    benign_breast_disease: bool = Field(..., description="History of benign breast disease")
    breast_biopsies: int = Field(..., ge=0, le=10, description="Number of breast biopsies")
    hormone_therapy_current: bool = Field(..., description="Currently using hormone therapy")
    hormone_therapy_years: Optional[float] = Field(None, ge=0, le=40, description="Years of hormone therapy")
    hormone_therapy_type: Optional[str] = Field(None, description="Type of hormone therapy")
    aspirin_use: Optional[bool] = Field(None, description="Regular aspirin use")
    ibuprofen_use: Optional[bool] = Field(None, description="Regular ibuprofen use")


class LifestyleFactors(BaseModel):
    """Lifestyle and behavioral factors."""
    smoking_status: str = Field(..., description="never, current, or former")
    pack_years: Optional[float] = Field(None, ge=0, le=100, description="Pack-years if smoker")
    alcohol_drinks_per_week: Optional[int] = Field(None, ge=0, le=50, description="Alcoholic drinks per week")
    physical_activity_hours: Optional[float] = Field(None, ge=0, le=50, description="Exercise hours per week")
    birth_control_ever: Optional[bool] = Field(None, description="Ever used birth control pills")
    birth_control_years: Optional[float] = Field(None, ge=0, le=40, description="Years of birth control use")
    
    @validator('smoking_status')
    def validate_smoking(cls, v):
        valid_status = ['never', 'current', 'former']
        if v.lower() not in valid_status:
            raise ValueError(f"Smoking status must be one of: {', '.join(valid_status)}")
        return v.lower()


class PredictionRequest(BaseModel):
    """Complete prediction request."""
    patient_id: Optional[str] = Field(None, description="Optional patient identifier")
    demographics: Demographics
    reproductive_history: ReproductiveHistory
    body_metrics: BodyMetrics
    medical_history: MedicalHistory
    lifestyle: Optional[LifestyleFactors] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "12345",
                "demographics": {
                    "age": 55,
                    "race": "white"
                },
                "reproductive_history": {
                    "age_at_menarche": 12,
                    "age_at_first_birth": 28,
                    "number_of_live_births": 2,
                    "first_degree_bc": 1
                },
                "body_metrics": {
                    "current_bmi": 26.5,
                    "height_cm": 165,
                    "weight_kg": 72
                },
                "medical_history": {
                    "personal_cancer_history": False,
                    "benign_breast_disease": True,
                    "breast_biopsies": 1,
                    "hormone_therapy_current": False,
                    "aspirin_use": True
                }
            }
        }


# Output schemas
class FeatureImportance(BaseModel):
    """Feature importance information."""
    feature: str
    importance: float = Field(..., ge=0, le=1)
    value: float
    contribution: float
    description: str


class ModelPrediction(BaseModel):
    """Individual model prediction."""
    risk_score: float = Field(..., ge=0, le=1)
    features_used: int
    model_version: str
    inference_time_ms: Optional[float] = None


class ModelComparison(BaseModel):
    """Comparison between different models."""
    ga_model: ModelPrediction
    baseline_model: ModelPrediction
    agreement: float = Field(..., ge=0, le=1, description="Agreement score between models")
    recommended_model: str


class ClinicalRecommendation(BaseModel):
    """Clinical recommendation."""
    type: RecommendationType
    priority: str = Field(..., description="high, medium, or low")
    action: str
    rationale: str
    potential_impact: Optional[str] = None


class ScreeningRecommendation(BaseModel):
    """Screening recommendation details."""
    recommendation: str
    frequency: str
    next_date: Optional[datetime] = None
    additional_imaging: Optional[List[str]] = None
    rationale: str


class PredictionResponse(BaseModel):
    """Complete prediction response."""
    prediction_id: str
    risk_score: float = Field(..., ge=0, le=1, description="5-year breast cancer risk")
    risk_category: RiskCategory
    confidence_interval: Dict[str, float]
    percentile: int = Field(..., ge=0, le=100, description="Risk percentile in population")
    relative_risk: Optional[float] = Field(None, description="Relative risk compared to average women of same age")
    feature_importance: List[FeatureImportance]
    model_comparison: ModelComparison
    recommendations: Optional[List[ClinicalRecommendation]] = None
    screening: Optional[ScreeningRecommendation] = None
    model_version: str
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "pred_123456",
                "risk_score": 0.183,
                "risk_category": "moderate",
                "confidence_interval": {"lower": 0.156, "upper": 0.210},
                "percentile": 85,
                "feature_importance": [
                    {
                        "feature": "family_history",
                        "importance": 0.15,
                        "value": 1,
                        "contribution": 0.045,
                        "description": "First-degree relative with breast cancer"
                    }
                ],
                "model_comparison": {
                    "ga_model": {
                        "risk_score": 0.183,
                        "features_used": 28,
                        "model_version": "1.2"
                    },
                    "baseline_model": {
                        "risk_score": 0.179,
                        "features_used": 92,
                        "model_version": "1.2"
                    },
                    "agreement": 0.96,
                    "recommended_model": "ga_model"
                },
                "model_version": "1.2",
                "processing_time_ms": 145.3,
                "timestamp": "2024-12-23T10:30:00Z"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    requests: List[PredictionRequest]
    
    @validator('requests')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 requests")
        return v


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    failed_requests: List[Dict[str, str]] = []
    total_requests: int
    successful_predictions: int
    processing_time_ms: float


# MLflow integration schemas
class ModelMetadata(BaseModel):
    """Model metadata from MLflow."""
    name: str
    version: str
    stage: str
    run_id: str
    created_at: int
    updated_at: int
    tags: Dict[str, str]


class ModelVersion(BaseModel):
    """Model version information."""
    name: str
    version: str
    stage: str
    run_id: str


class ExperimentInfo(BaseModel):
    """MLflow experiment information."""
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str
    tags: Dict[str, str]


class RunInfo(BaseModel):
    """MLflow run information."""
    run_id: str
    experiment_id: str
    status: str
    start_time: int
    end_time: Optional[int]
    metrics: Dict[str, float]
    params: Dict[str, str]
    tags: Dict[str, str]


# Health check schemas
class HealthStatus(BaseModel):
    """Health check status."""
    status: str = Field(..., description="healthy, degraded, or unhealthy")
    timestamp: datetime
    uptime_seconds: float
    version: str


class ModelHealth(BaseModel):
    """Model health information."""
    model_name: str
    loaded: bool
    version: str
    last_prediction: Optional[datetime]
    prediction_count: int
    average_latency_ms: Optional[float]
    error_rate: Optional[float]


class SystemHealth(BaseModel):
    """Complete system health."""
    api: HealthStatus
    models: List[ModelHealth]
    database: Dict[str, Any]
    mlflow: Dict[str, Any]


# Analytics schemas
class PredictionStats(BaseModel):
    """Prediction statistics."""
    total_predictions: int
    average_risk_score: float
    risk_distribution: Dict[str, int]
    average_latency_ms: float
    error_count: int
    time_period: str


class FeatureDistribution(BaseModel):
    """Feature distribution statistics."""
    feature: str
    mean: float
    std: float
    min: float
    max: float
    missing_count: int


class ModelPerformance(BaseModel):
    """Model performance metrics."""
    model_name: str
    auc: float
    precision: float
    recall: float
    f1_score: float
    evaluation_date: datetime
    test_size: int


# Error schemas
class ErrorResponse(BaseModel):
    """Error response format."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
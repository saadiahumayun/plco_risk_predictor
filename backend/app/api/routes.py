# app/api/routes.py
"""
API route definitions for the breast cancer risk prediction service.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session

from app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelHealth,
    SystemHealth,
    PredictionStats,
    ModelPerformance,
    ErrorResponse
)
from app.services.ml_service import ml_service
from app.services.mlflow_service import mlflow_service
from app.services.monitoring import monitoring_service
from app.services.analytics import analytics_service
from app.core.security import get_current_user, get_current_user_optional
from app.core.config import settings
from app.utils.validators import validate_prediction_request
from app.db.session import get_db
from app.db.models import Prediction


logger = logging.getLogger(__name__)


DEMO_USER_ID = "00000000-0000-0000-0000-000000000001"  # Demo user for unauthenticated predictions


def save_prediction_to_db(
    request_data: dict,
    response_data: dict,
    user_id: Optional[str] = None
):
    """Save a prediction to the database."""
    # Skip database save in demo mode
    if settings.ENVIRONMENT in ("demo", "test"):
        logger.info(f"Demo mode: Skipping database save for prediction {response_data.get('prediction_id')}")
        return
    
    from app.db.session import SessionLocal
    
    db = SessionLocal()
    try:
        # Extract patient_id (MR number) from request
        patient_id = request_data.get("patient_id") or request_data.get("mr_number")
        
        # Use demo user ID if no user is authenticated
        effective_user_id = user_id or DEMO_USER_ID
        
        prediction = Prediction(
            prediction_id=response_data["prediction_id"],
            user_id=effective_user_id,
            patient_id=patient_id,  # MR Number
            risk_score=response_data["risk_score"],
            risk_category=response_data["risk_category"].value if hasattr(response_data["risk_category"], 'value') else response_data["risk_category"],
            confidence_lower=response_data.get("confidence_interval", {}).get("lower") if response_data.get("confidence_interval") else None,
            confidence_upper=response_data.get("confidence_interval", {}).get("upper") if response_data.get("confidence_interval") else None,
            percentile=response_data.get("percentile"),
            relative_risk=response_data.get("relative_risk"),
            model_name=settings.GA_MODEL_NAME,
            model_version=response_data.get("model_version") or "unknown",
            model_type="ga_optimized",
            input_features=request_data,
            feature_importance=response_data.get("feature_importance"),
            comparison_results=response_data.get("model_comparison"),
            processing_time_ms=response_data.get("processing_time_ms"),
        )
        db.add(prediction)
        db.commit()
        logger.info(f"Saved prediction {response_data['prediction_id']} for patient {patient_id} to database")
    except Exception as e:
        logger.error(f"Failed to save prediction to database: {e}")
        db.rollback()
    finally:
        db.close()

# Create main API router
api_router = APIRouter()


# Prediction endpoints
@api_router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Make a breast cancer risk prediction",
    description="Predict 5-year breast cancer risk using GA-optimized model"
)
async def predict_risk(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """Make a single risk prediction."""
    try:
        # Validate request
        validation_errors = validate_prediction_request(request)
        if validation_errors:
            raise HTTPException(status_code=422, detail=validation_errors)
        
        # Make prediction
        response = ml_service.predict(request)
        
        # Save prediction to database in background
        user_id = current_user.get("user_id") if current_user else None
        # Serialize data for background task (to avoid session issues)
        request_data = request.model_dump() if hasattr(request, 'model_dump') else request.dict()
        response_data = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
        background_tasks.add_task(
            save_prediction_to_db,
            request_data=request_data,
            response_data=response_data,
            user_id=user_id
        )
        
        # Track analytics in background
        background_tasks.add_task(
            analytics_service.track_prediction,
            prediction_id=response.prediction_id,
            risk_score=response.risk_score,
            risk_category=response.risk_category,
            user_id=user_id
        )
        
        # Monitor prediction
        monitoring_service.record_prediction(
            risk_score=response.risk_score,
            latency_ms=response.processing_time_ms,
            model_version=response.model_version
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Make batch predictions",
    description="Process multiple predictions in a single request"
)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """Make batch predictions."""
    try:
        start_time = datetime.utcnow()
        
        # Process predictions
        predictions = []
        failed_requests = []
        
        for idx, pred_request in enumerate(request.requests):
            try:
                response = ml_service.predict(pred_request)
                predictions.append(response)
            except Exception as e:
                failed_requests.append({
                    "index": idx,
                    "error": str(e)
                })
        
        # Calculate processing time
        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Track batch analytics
        background_tasks.add_task(
            analytics_service.track_batch_prediction,
            total_requests=len(request.requests),
            successful=len(predictions),
            failed=len(failed_requests),
            user_id=current_user.get("user_id") if current_user else None
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            failed_requests=failed_requests,
            total_requests=len(request.requests),
            successful_predictions=len(predictions),
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Prediction history endpoints
@api_router.get(
    "/predictions",
    summary="Get prediction history",
    description="Get list of past predictions"
)
async def get_predictions(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """Get prediction history."""
    try:
        query = db.query(Prediction).order_by(Prediction.created_at.desc())
        
        # If user is authenticated, show their predictions
        if current_user and current_user.get("user_id"):
            query = query.filter(Prediction.user_id == current_user["user_id"])
        
        total = query.count()
        predictions = query.offset(skip).limit(limit).all()
        
        return {
            "predictions": [
                {
                    "prediction_id": p.prediction_id,
                    "risk_score": p.risk_score,
                    "risk_category": p.risk_category,
                    "percentile": p.percentile,
                    "model_version": p.model_version,
                    "processing_time_ms": p.processing_time_ms,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "confidence_interval": {
                        "lower": p.confidence_lower,
                        "upper": p.confidence_upper
                    } if p.confidence_lower and p.confidence_upper else None
                }
                for p in predictions
            ],
            "total": total,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get(
    "/predictions/{prediction_id}",
    summary="Get single prediction",
    description="Get details of a specific prediction"
)
async def get_prediction(
    prediction_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """Get a specific prediction by ID."""
    try:
        prediction = db.query(Prediction).filter(
            Prediction.prediction_id == prediction_id
        ).first()
        
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return {
            "prediction_id": prediction.prediction_id,
            "risk_score": prediction.risk_score,
            "risk_category": prediction.risk_category,
            "percentile": prediction.percentile,
            "confidence_interval": {
                "lower": prediction.confidence_lower,
                "upper": prediction.confidence_upper
            } if prediction.confidence_lower and prediction.confidence_upper else None,
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
            "input_features": prediction.input_features,
            "feature_importance": prediction.feature_importance,
            "processing_time_ms": prediction.processing_time_ms,
            "created_at": prediction.created_at.isoformat() if prediction.created_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model management endpoints
@api_router.get(
    "/models/info",
    summary="Get model information",
    description="Get information about loaded models"
)
async def get_model_info():
    """Get information about loaded models."""
    try:
        return ml_service.get_model_info()
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post(
    "/models/reload",
    summary="Reload models",
    description="Reload models from MLflow (requires authentication)"
)
async def reload_models(
    current_user: Dict = Depends(get_current_user)
):
    """Reload models from MLflow."""
    try:
        # Check if user has admin privileges
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        result = ml_service.reload_models()
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reload models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get(
    "/models/{model_name}/metrics",
    summary="Get model metrics",
    description="Get performance metrics for a specific model"
)
async def get_model_metrics(
    model_name: str,
    version: Optional[str] = None
):
    """Get model metrics from MLflow."""
    try:
        metrics = mlflow_service.get_model_metrics(model_name, version)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get(
    "/models/{model_name}/lineage",
    summary="Get model lineage",
    description="Get lineage information for a model"
)
async def get_model_lineage(
    model_name: str,
    version: Optional[str] = None
):
    """Get model lineage information."""
    try:
        lineage = mlflow_service.get_model_lineage(model_name, version)
        return lineage
    except Exception as e:
        logger.error(f"Failed to get model lineage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# MLflow integration endpoints
@api_router.get(
    "/mlflow/experiments",
    summary="List MLflow experiments",
    description="Get list of MLflow experiments"
)
async def list_experiments(
    query: Optional[str] = Query(None, description="Filter query")
):
    """List MLflow experiments."""
    try:
        experiments = mlflow_service.search_experiments(query)
        return {"experiments": experiments}
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get(
    "/mlflow/runs",
    summary="Search MLflow runs",
    description="Search for MLflow runs"
)
async def search_runs(
    experiment_id: str,
    filter_string: Optional[str] = Query("", description="MLflow filter string"),
    max_results: int = Query(100, le=1000)
):
    """Search MLflow runs."""
    try:
        runs = mlflow_service.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=max_results
        )
        return {"runs": runs}
    except Exception as e:
        logger.error(f"Failed to search runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health and monitoring endpoints
@api_router.get(
    "/health/detailed",
    response_model=SystemHealth,
    summary="Detailed health check",
    description="Get detailed system health information"
)
async def detailed_health():
    """Get detailed system health."""
    try:
        health = monitoring_service.get_system_health()
        return health
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get(
    "/health/models",
    response_model=List[ModelHealth],
    summary="Model health",
    description="Get health status of all models"
)
async def model_health():
    """Get model health information."""
    try:
        health = monitoring_service.get_model_health()
        return health
    except Exception as e:
        logger.error(f"Failed to get model health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics endpoints
@api_router.get(
    "/analytics/predictions",
    response_model=PredictionStats,
    summary="Prediction statistics",
    description="Get prediction statistics"
)
async def prediction_stats(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    time_period: str = Query("24h", description="Time period: 1h, 24h, 7d, 30d")
):
    """Get prediction statistics."""
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            if time_period == "1h":
                start_date = end_date - timedelta(hours=1)
            elif time_period == "24h":
                start_date = end_date - timedelta(days=1)
            elif time_period == "7d":
                start_date = end_date - timedelta(days=7)
            elif time_period == "30d":
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=1)
        
        stats = analytics_service.get_prediction_stats(start_date, end_date)
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get prediction stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get(
    "/analytics/performance",
    response_model=List[ModelPerformance],
    summary="Model performance",
    description="Get model performance metrics over time"
)
async def model_performance(
    model_name: Optional[str] = None,
    days: int = Query(30, description="Number of days to look back")
):
    """Get model performance metrics."""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        performance = analytics_service.get_model_performance(
            model_name=model_name,
            start_date=start_date
        )
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get(
    "/analytics/drift",
    summary="Data drift analysis",
    description="Get data drift analysis results"
)
async def data_drift(
    feature: Optional[str] = Query(None, description="Specific feature to analyze"),
    days: int = Query(7, description="Number of days to analyze")
):
    """Get data drift analysis."""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        drift_results = analytics_service.analyze_drift(
            feature=feature,
            start_date=start_date
        )
        return drift_results
        
    except Exception as e:
        logger.error(f"Failed to analyze drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Feature information endpoint
@api_router.get(
    "/features",
    summary="Get feature information",
    description="Get information about model features"
)
async def get_features():
    """Get feature information."""
    return {
        "ga_features": {
            "count": len(ml_service.ga_features),
            "features": ml_service.ga_features,
            "description": "Features selected by genetic algorithm"
        },
        "baseline_features": {
            "count": len(ml_service.baseline_features),
            "features": ml_service.baseline_features,
            "description": "All available features"
        },
        "clinical_groups": CLINICAL_FEATURES
    }


# Validation endpoint
@api_router.post(
    "/validate",
    summary="Validate prediction input",
    description="Validate prediction input without making a prediction"
)
async def validate_input(request: PredictionRequest):
    """Validate prediction input."""
    try:
        errors = validate_prediction_request(request)
        if errors:
            return {"valid": False, "errors": errors}
        return {"valid": True, "errors": []}
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}


# Error handling
@api_router.get("/error-test", include_in_schema=False)
async def error_test():
    """Test error handling (development only)."""
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    raise Exception("This is a test error")
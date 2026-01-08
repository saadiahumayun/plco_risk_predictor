# app/services/analytics.py
"""
Analytics service for tracking and analyzing predictions.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.schemas import PredictionStats, FeatureDistribution, ModelPerformance
from app.db.session import get_db


logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for analytics and reporting."""
    
    def __init__(self):
        self.prediction_cache = []
        self.batch_cache = []
        self.feature_cache = defaultdict(list)
        
    async def track_prediction(self, 
                              prediction_id: str,
                              risk_score: float,
                              risk_category: str,
                              user_id: Optional[str] = None,
                              features: Optional[Dict[str, Any]] = None):
        """Track a prediction for analytics."""
        try:
            prediction_data = {
                "prediction_id": prediction_id,
                "risk_score": risk_score,
                "risk_category": risk_category,
                "user_id": user_id,
                "timestamp": datetime.utcnow(),
                "features": features
            }
            
            # In production, save to database
            self.prediction_cache.append(prediction_data)
            
            # Track features for distribution analysis
            if features:
                for feature, value in features.items():
                    if isinstance(value, (int, float)):
                        self.feature_cache[feature].append(value)
            
            # Periodic flush to database (in production)
            if len(self.prediction_cache) >= 100:
                await self._flush_to_database()
                
        except Exception as e:
            logger.error(f"Failed to track prediction: {e}")
    
    async def track_batch_prediction(self,
                                   total_requests: int,
                                   successful: int,
                                   failed: int,
                                   user_id: Optional[str] = None):
        """Track batch prediction metrics."""
        try:
            batch_data = {
                "total_requests": total_requests,
                "successful": successful,
                "failed": failed,
                "user_id": user_id,
                "timestamp": datetime.utcnow()
            }
            
            self.batch_cache.append(batch_data)
            
        except Exception as e:
            logger.error(f"Failed to track batch prediction: {e}")
    
    def get_prediction_stats(self, 
                           start_date: datetime,
                           end_date: datetime) -> PredictionStats:
        """Get prediction statistics for a time period."""
        try:
            # Filter predictions in date range
            predictions = [
                p for p in self.prediction_cache
                if start_date <= p["timestamp"] <= end_date
            ]
            
            if not predictions:
                return PredictionStats(
                    total_predictions=0,
                    average_risk_score=0.0,
                    risk_distribution={"low": 0, "moderate": 0, "high": 0},
                    average_latency_ms=0.0,
                    error_count=0,
                    time_period=f"{start_date.date()} to {end_date.date()}"
                )
            
            # Calculate statistics
            risk_scores = [p["risk_score"] for p in predictions]
            risk_distribution = defaultdict(int)
            for p in predictions:
                risk_distribution[p["risk_category"]] += 1
            
            return PredictionStats(
                total_predictions=len(predictions),
                average_risk_score=float(np.mean(risk_scores)),
                risk_distribution=dict(risk_distribution),
                average_latency_ms=150.0,  # Would come from monitoring service
                error_count=0,  # Would come from error tracking
                time_period=f"{start_date.date()} to {end_date.date()}"
            )
            
        except Exception as e:
            logger.error(f"Failed to get prediction stats: {e}")
            raise
    
    def get_feature_distributions(self, 
                                features: Optional[List[str]] = None) -> List[FeatureDistribution]:
        """Get feature distribution statistics."""
        try:
            distributions = []
            
            features_to_analyze = features or list(self.feature_cache.keys())
            
            for feature in features_to_analyze:
                if feature in self.feature_cache and self.feature_cache[feature]:
                    values = self.feature_cache[feature]
                    
                    dist = FeatureDistribution(
                        feature=feature,
                        mean=float(np.mean(values)),
                        std=float(np.std(values)),
                        min=float(np.min(values)),
                        max=float(np.max(values)),
                        missing_count=0  # Would track actual missing values
                    )
                    distributions.append(dist)
            
            return distributions
            
        except Exception as e:
            logger.error(f"Failed to get feature distributions: {e}")
            return []
    
    def get_model_performance(self,
                            model_name: Optional[str] = None,
                            start_date: Optional[datetime] = None) -> List[ModelPerformance]:
        """Get model performance metrics over time."""
        try:
            # In production, this would query actual evaluation results
            # For now, return mock data
            performances = []
            
            if model_name == "ga_model" or model_name is None:
                performances.append(ModelPerformance(
                    model_name="ga_model",
                    auc=0.892,
                    precision=0.85,
                    recall=0.78,
                    f1_score=0.81,
                    evaluation_date=datetime.utcnow(),
                    test_size=891
                ))
            
            if model_name == "baseline_model" or model_name is None:
                performances.append(ModelPerformance(
                    model_name="baseline_model",
                    auc=0.876,
                    precision=0.82,
                    recall=0.75,
                    f1_score=0.78,
                    evaluation_date=datetime.utcnow(),
                    test_size=891
                ))
            
            return performances
            
        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            return []
    
    def analyze_drift(self,
                     feature: Optional[str] = None,
                     start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Analyze data drift for features."""
        try:
            drift_analysis = {
                "analysis_date": datetime.utcnow().isoformat(),
                "features_analyzed": [],
                "drift_detected": False,
                "details": {}
            }
            
            features_to_analyze = [feature] if feature else list(self.feature_cache.keys())
            
            for feat in features_to_analyze:
                if feat in self.feature_cache and len(self.feature_cache[feat]) >= 100:
                    values = self.feature_cache[feat]
                    
                    # Simple drift detection - compare first half vs second half
                    mid = len(values) // 2
                    first_half = values[:mid]
                    second_half = values[mid:]
                    
                    # T-test for mean shift (simplified)
                    mean_shift = abs(np.mean(first_half) - np.mean(second_half))
                    std_pooled = np.sqrt((np.std(first_half)**2 + np.std(second_half)**2) / 2)
                    
                    if std_pooled > 0:
                        drift_score = mean_shift / std_pooled
                    else:
                        drift_score = 0.0
                    
                    drift_analysis["details"][feat] = {
                        "drift_score": float(drift_score),
                        "mean_before": float(np.mean(first_half)),
                        "mean_after": float(np.mean(second_half)),
                        "samples": len(values),
                        "drift_detected": drift_score > 0.1
                    }
                    
                    if drift_score > 0.1:
                        drift_analysis["drift_detected"] = True
                    
                    drift_analysis["features_analyzed"].append(feat)
            
            return drift_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze drift: {e}")
            return {"error": str(e)}
    
    async def _flush_to_database(self):
        """Flush cached predictions to database."""
        try:
            # In production, bulk insert to database
            logger.info(f"Flushing {len(self.prediction_cache)} predictions to database")
            
            # Clear cache
            self.prediction_cache = []
            
        except Exception as e:
            logger.error(f"Failed to flush to database: {e}")
    
    def get_risk_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get risk score trends over time."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Filter predictions
            predictions = [
                p for p in self.prediction_cache
                if start_date <= p["timestamp"] <= end_date
            ]
            
            if not predictions:
                return {"error": "No predictions in time range"}
            
            # Group by day
            df = pd.DataFrame(predictions)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            
            daily_stats = df.groupby('date').agg({
                'risk_score': ['mean', 'std', 'count'],
                'risk_category': lambda x: x.value_counts().to_dict()
            }).to_dict()
            
            return {
                "period_days": days,
                "daily_statistics": daily_stats,
                "overall_trend": "stable"  # Would calculate actual trend
            }
            
        except Exception as e:
            logger.error(f"Failed to get risk trends: {e}")
            return {"error": str(e)}
    
    def get_feature_importance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Track how feature importance changes over time."""
        try:
            # This would integrate with MLflow to track feature importance over multiple model versions
            return {
                "period_days": days,
                "top_features_stable": ["age", "bmi_curr", "fmenstr", "fh_cancer"],
                "emerging_features": ["lifestyle_score", "genetic_markers"],
                "declining_features": ["occupation", "marital_status"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get feature importance trends: {e}")
            return {"error": str(e)}


# Create singleton instance
analytics_service = AnalyticsService()
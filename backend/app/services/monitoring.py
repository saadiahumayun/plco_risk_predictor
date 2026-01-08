# app/services/monitoring.py
"""
Monitoring service for tracking model performance and system health.
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict, deque
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Info
import asyncio

from app.core.config import settings
from app.models.schemas import ModelHealth, SystemHealth, HealthStatus


logger = logging.getLogger(__name__)


# Prometheus metrics
prediction_counter = Counter(
    'predictions_total', 
    'Total number of predictions',
    ['model', 'risk_category']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model']
)

risk_score_gauge = Gauge(
    'risk_score_current',
    'Current risk score',
    ['model']
)

model_info = Info(
    'model_info',
    'Model information',
    ['model']
)

feature_drift_gauge = Gauge(
    'feature_drift_score',
    'Feature drift score',
    ['feature']
)

error_counter = Counter(
    'prediction_errors_total',
    'Total number of prediction errors',
    ['model', 'error_type']
)


class MonitoringService:
    """Service for monitoring model performance and system health."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.prediction_history = defaultdict(lambda: deque(maxlen=1000))
        self.latency_history = defaultdict(lambda: deque(maxlen=1000))
        self.error_history = defaultdict(list)
        self.feature_distributions = defaultdict(lambda: deque(maxlen=1000))
        self.last_prediction_time = {}
        self.model_prediction_counts = defaultdict(int)
        self.risk_distribution = defaultdict(lambda: defaultdict(int))
        
        # Drift detection
        self.baseline_distributions = {}
        self.drift_scores = defaultdict(float)
        self.drift_check_counter = 0
        
        # Performance metrics
        self.performance_window = deque(maxlen=100)
        
        logger.info("Monitoring service initialized")
    
    def start(self):
        """Start monitoring background tasks."""
        # In production, you might use Celery or similar for background tasks
        logger.info("Monitoring service started")
    
    def stop(self):
        """Stop monitoring tasks."""
        logger.info("Monitoring service stopped")
    
    def record_prediction(self, 
                         risk_score: float,
                         latency_ms: float,
                         model_version: str,
                         risk_category: str = "unknown",
                         features: Optional[Dict[str, Any]] = None):
        """Record a prediction for monitoring."""
        try:
            # Update counters
            prediction_counter.labels(
                model=model_version,
                risk_category=risk_category
            ).inc()
            
            # Record latency
            prediction_latency.labels(model=model_version).observe(latency_ms / 1000.0)
            
            # Update risk score gauge
            risk_score_gauge.labels(model=model_version).set(risk_score)
            
            # Store in history
            self.prediction_history[model_version].append({
                'risk_score': risk_score,
                'risk_category': risk_category,
                'timestamp': datetime.utcnow(),
                'latency_ms': latency_ms
            })
            
            self.latency_history[model_version].append(latency_ms)
            self.last_prediction_time[model_version] = datetime.utcnow()
            self.model_prediction_counts[model_version] += 1
            self.risk_distribution[model_version][risk_category] += 1
            
            # Check for drift periodically
            self.drift_check_counter += 1
            if self.drift_check_counter >= settings.DRIFT_CHECK_FREQUENCY:
                self.check_drift(features)
                self.drift_check_counter = 0
            
            # Store features for drift detection
            if features:
                for feature, value in features.items():
                    if isinstance(value, (int, float)):
                        self.feature_distributions[feature].append(value)
            
        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")
    
    def record_error(self, model_version: str, error_type: str, error_message: str):
        """Record a prediction error."""
        try:
            error_counter.labels(
                model=model_version,
                error_type=error_type
            ).inc()
            
            self.error_history[model_version].append({
                'error_type': error_type,
                'error_message': error_message,
                'timestamp': datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Failed to record error: {e}")
    
    def check_drift(self, current_features: Optional[Dict[str, Any]] = None):
        """Check for feature drift."""
        if not current_features:
            return
        
        try:
            for feature, value in current_features.items():
                if isinstance(value, (int, float)) and feature in self.feature_distributions:
                    recent_values = list(self.feature_distributions[feature])
                    if len(recent_values) >= 100:
                        # Simple drift detection using KL divergence approximation
                        drift_score = self._calculate_drift_score(
                            recent_values,
                            self.baseline_distributions.get(feature, recent_values[:50])
                        )
                        self.drift_scores[feature] = drift_score
                        feature_drift_gauge.labels(feature=feature).set(drift_score)
                        
                        if drift_score > 0.1:  # Threshold for drift alert
                            logger.warning(f"Feature drift detected for {feature}: {drift_score}")
            
        except Exception as e:
            logger.error(f"Failed to check drift: {e}")
    
    def _calculate_drift_score(self, current: List[float], baseline: List[float]) -> float:
        """Calculate drift score between two distributions."""
        try:
            # Simple statistical test - you might want to use more sophisticated methods
            current_mean = np.mean(current)
            current_std = np.std(current)
            baseline_mean = np.mean(baseline)
            baseline_std = np.std(baseline)
            
            # Normalized difference
            if baseline_std > 0:
                z_score = abs(current_mean - baseline_mean) / baseline_std
                return min(z_score / 10.0, 1.0)  # Normalize to 0-1
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to calculate drift score: {e}")
            return 0.0
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health status."""
        try:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            # API health
            api_health = HealthStatus(
                status="healthy",
                timestamp=datetime.utcnow(),
                uptime_seconds=uptime,
                version=settings.VERSION
            )
            
            # Model health
            model_healths = []
            for model_name in ["ga_model", "baseline_model"]:
                model_health = self.get_model_health_status(model_name)
                model_healths.append(model_health)
            
            # Database health (simplified)
            db_health = {
                "status": "healthy",
                "connection_pool_size": 10,
                "active_connections": 3
            }
            
            # MLflow health
            mlflow_health = {
                "status": "healthy",
                "tracking_uri": settings.MLFLOW_TRACKING_URI,
                "experiment": settings.MLFLOW_EXPERIMENT_NAME
            }
            
            return SystemHealth(
                api=api_health,
                models=model_healths,
                database=db_health,
                mlflow=mlflow_health
            )
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return SystemHealth(
                api=HealthStatus(
                    status="unhealthy",
                    timestamp=datetime.utcnow(),
                    uptime_seconds=0,
                    version=settings.VERSION
                ),
                models=[],
                database={"status": "unknown"},
                mlflow={"status": "unknown"}
            )
    
    def get_model_health_status(self, model_name: str) -> ModelHealth:
        """Get health status for a specific model."""
        try:
            # Calculate metrics
            prediction_count = self.model_prediction_counts.get(model_name, 0)
            last_prediction = self.last_prediction_time.get(model_name)
            
            # Average latency
            latencies = list(self.latency_history.get(model_name, []))
            avg_latency = np.mean(latencies) if latencies else None
            
            # Error rate
            total_errors = len(self.error_history.get(model_name, []))
            error_rate = total_errors / max(prediction_count, 1)
            
            # Determine if model is loaded (simplified check)
            loaded = prediction_count > 0 or model_name in self.last_prediction_time
            
            return ModelHealth(
                model_name=model_name,
                loaded=loaded,
                version="1.0",  # Should get from ml_service
                last_prediction=last_prediction,
                prediction_count=prediction_count,
                average_latency_ms=avg_latency,
                error_rate=error_rate
            )
            
        except Exception as e:
            logger.error(f"Failed to get model health: {e}")
            return ModelHealth(
                model_name=model_name,
                loaded=False,
                version="unknown",
                last_prediction=None,
                prediction_count=0,
                average_latency_ms=None,
                error_rate=None
            )
    
    def get_model_health(self) -> List[ModelHealth]:
        """Get health status for all models."""
        models = [
            self.get_model_health_status("ga_model"),
            self.get_model_health_status("baseline_model")
        ]
        return models
    
    def get_performance_metrics(self, model_name: str, window_hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for a model over a time window."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
            
            # Filter predictions within window
            recent_predictions = [
                p for p in self.prediction_history.get(model_name, [])
                if p['timestamp'] > cutoff_time
            ]
            
            if not recent_predictions:
                return {
                    "model_name": model_name,
                    "window_hours": window_hours,
                    "prediction_count": 0,
                    "average_risk_score": None,
                    "risk_distribution": {},
                    "average_latency_ms": None
                }
            
            # Calculate metrics
            risk_scores = [p['risk_score'] for p in recent_predictions]
            latencies = [p['latency_ms'] for p in recent_predictions]
            
            # Risk distribution
            risk_dist = defaultdict(int)
            for p in recent_predictions:
                risk_dist[p['risk_category']] += 1
            
            return {
                "model_name": model_name,
                "window_hours": window_hours,
                "prediction_count": len(recent_predictions),
                "average_risk_score": np.mean(risk_scores),
                "risk_score_std": np.std(risk_scores),
                "risk_distribution": dict(risk_dist),
                "average_latency_ms": np.mean(latencies),
                "latency_p95": np.percentile(latencies, 95) if latencies else None,
                "error_count": len([e for e in self.error_history.get(model_name, []) 
                                   if e['timestamp'] > cutoff_time])
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    def get_drift_report(self) -> Dict[str, Any]:
        """Get feature drift report."""
        try:
            drift_report = {
                "timestamp": datetime.utcnow().isoformat(),
                "features": {}
            }
            
            for feature, score in self.drift_scores.items():
                drift_report["features"][feature] = {
                    "drift_score": score,
                    "status": "drifted" if score > 0.1 else "stable",
                    "sample_size": len(self.feature_distributions[feature])
                }
            
            # Overall drift score
            if self.drift_scores:
                drift_report["overall_drift_score"] = np.mean(list(self.drift_scores.values()))
                drift_report["drifted_features"] = [
                    f for f, s in self.drift_scores.items() if s > 0.1
                ]
            else:
                drift_report["overall_drift_score"] = 0.0
                drift_report["drifted_features"] = []
            
            return drift_report
            
        except Exception as e:
            logger.error(f"Failed to get drift report: {e}")
            return {"error": str(e)}
    
    def should_retrain(self) -> bool:
        """Determine if models should be retrained based on monitoring metrics."""
        try:
            # Check multiple conditions
            conditions = []
            
            # 1. Performance degradation
            for model_name in ["ga_model", "baseline_model"]:
                metrics = self.get_performance_metrics(model_name, window_hours=168)  # 1 week
                if metrics.get("prediction_count", 0) > 100:
                    # Check if error rate is high
                    error_rate = metrics.get("error_count", 0) / metrics["prediction_count"]
                    if error_rate > 0.05:  # 5% error rate
                        conditions.append(f"High error rate for {model_name}: {error_rate:.2%}")
            
            # 2. Feature drift
            drift_report = self.get_drift_report()
            if drift_report.get("overall_drift_score", 0) > 0.15:
                conditions.append(f"High overall drift score: {drift_report['overall_drift_score']:.3f}")
            
            # 3. Time-based retraining
            # This would check when the model was last trained
            # For now, we'll skip this check
            
            if conditions:
                logger.warning(f"Retraining recommended: {', '.join(conditions)}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check retraining conditions: {e}")
            return False


# Create singleton instance
monitoring_service = MonitoringService()
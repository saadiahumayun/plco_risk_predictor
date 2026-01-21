# app/services/mlflow_service.py
"""
MLflow integration service for model loading, tracking, and management.
MLflow is optional - if not installed, service runs in stub mode.
"""
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import json

from app.core.config import settings
from app.models.schemas import ModelMetadata, ModelVersion

logger = logging.getLogger(__name__)

# Try to import mlflow - it's optional
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient
    from tenacity import retry, stop_after_attempt, wait_exponential
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed - running in stub mode (pickle file loading only)")


class MLflowService:
    """Service for managing MLflow operations."""
    
    def __init__(self):
        self.tracking_uri = settings.MLFLOW_TRACKING_URI
        self.experiment_name = settings.MLFLOW_EXPERIMENT_NAME
        
        # Cache for loaded models
        self._model_cache = {}
        
        # Model metadata
        self.ga_model_metadata = None
        self.baseline_model_metadata = None
        
        if MLFLOW_AVAILABLE:
            try:
                # Set tracking URI
                mlflow.set_tracking_uri(self.tracking_uri)
                
                # Create MLflow client
                self.client = MlflowClient(tracking_uri=self.tracking_uri)
                
                # Set or create experiment
                self._setup_experiment()
                
                logger.info(f"MLflow service initialized with tracking URI: {self.tracking_uri}")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}")
                self.client = None
        else:
            self.client = None
            logger.info("MLflow service running in stub mode (no MLflow installed)")
    
    def _setup_experiment(self):
        """Set up MLflow experiment."""
        if not MLFLOW_AVAILABLE or not self.client:
            self.experiment_id = None
            return
            
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
            else:
                self.experiment_id = self.client.create_experiment(self.experiment_name)
            
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"Using MLflow experiment: {self.experiment_name} (ID: {self.experiment_id})")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow experiment: {e}")
            logger.info("MLflow is not available - running in demo mode")
            self.experiment_id = None
    
    def load_model(self, model_name: str, version: Optional[str] = None, stage: Optional[str] = None) -> Any:
        """
        Load a model from MLflow registry.
        
        Args:
            model_name: Name of the model in registry
            version: Specific version to load
            stage: Model stage (Production, Staging, etc.)
        
        Returns:
            Loaded model
        """
        # If MLflow is not available, raise immediately so ml_service falls back to pickle
        if not MLFLOW_AVAILABLE or not self.client:
            raise RuntimeError("MLflow is not available - use pickle file loading instead")
        
        cache_key = f"{model_name}_{version or stage or 'latest'}"
        
        # Check cache first
        if cache_key in self._model_cache:
            logger.info(f"Loading model {cache_key} from cache")
            return self._model_cache[cache_key]
        
        try:
            if settings.USE_MODEL_REGISTRY:
                # Load from model registry
                if version:
                    model_uri = f"models:/{model_name}/{version}"
                elif stage and stage.strip():
                    model_uri = f"models:/{model_name}/{stage}"
                else:
                    # Load latest version (version 1)
                    model_uri = f"models:/{model_name}/1"
            else:
                # Load from run artifacts
                # This would require run_id which we'd get from somewhere
                raise NotImplementedError("Loading without registry not implemented yet")
            
            logger.info(f"Loading model from: {model_uri}")
            # Try sklearn flavor first, fallback to pyfunc
            try:
                model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Loaded model using sklearn flavor")
            except Exception as e:
                logger.warning(f"sklearn flavor failed, trying pyfunc: {e}")
                model = mlflow.pyfunc.load_model(model_uri)
            
            # Cache the model
            self._model_cache[cache_key] = model
            
            # Get model metadata
            self._load_model_metadata(model_name, version, stage)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _load_model_metadata(self, model_name: str, version: Optional[str], stage: Optional[str]):
        """Load metadata for a model."""
        if not MLFLOW_AVAILABLE or not self.client:
            return
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
            else:
                # Get latest version in stage
                versions = self.client.search_model_versions(
                    filter_string=f"name='{model_name}' and current_stage='{stage or settings.MODEL_STAGE}'"
                )
                if versions:
                    model_version = versions[0]
                else:
                    logger.warning(f"No model found for {model_name} in stage {stage}")
                    return
            
            # Extract metadata
            metadata = ModelMetadata(
                name=model_name,
                version=model_version.version,
                stage=model_version.current_stage,
                run_id=model_version.run_id,
                created_at=model_version.creation_timestamp,
                updated_at=model_version.last_updated_timestamp,
                tags=model_version.tags
            )
            
            # Store metadata
            if model_name == settings.GA_MODEL_NAME:
                self.ga_model_metadata = metadata
            elif model_name == settings.BASELINE_MODEL_NAME:
                self.baseline_model_metadata = metadata
                
            logger.info(f"Loaded metadata for {model_name} v{metadata.version}")
            
        except Exception as e:
            logger.error(f"Failed to load model metadata: {e}")
    
    def log_prediction(self, 
                      prediction_id: str,
                      model_name: str,
                      input_features: Dict[str, Any],
                      prediction: float,
                      risk_category: str,
                      latency_ms: float,
                      metadata: Optional[Dict[str, Any]] = None):
        """
        Log a prediction to MLflow tracking.
        
        Args:
            prediction_id: Unique prediction ID
            model_name: Name of the model used
            input_features: Input features used
            prediction: Risk score
            risk_category: Risk category (low/moderate/high)
            latency_ms: Prediction latency in milliseconds
            metadata: Additional metadata
        """
        if not settings.LOG_PREDICTIONS or not MLFLOW_AVAILABLE:
            return
        
        try:
            with mlflow.start_run(run_name=f"prediction_{prediction_id}"):
                # Log parameters
                mlflow.log_param("prediction_id", prediction_id)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("risk_category", risk_category)
                mlflow.log_param("timestamp", datetime.utcnow().isoformat())
                
                # Log metrics
                mlflow.log_metric("risk_score", prediction)
                mlflow.log_metric("latency_ms", latency_ms)
                
                # Log features as parameters (subset to avoid too many)
                important_features = ["age", "bmi_curr", "fmenstr", "fchilda", "race7", "fh_cancer"]
                for feature in important_features:
                    if feature in input_features:
                        mlflow.log_param(f"feature_{feature}", input_features[feature])
                
                # Log complete input as artifact
                input_json = json.dumps(input_features, indent=2)
                with open(f"/tmp/prediction_{prediction_id}_input.json", "w") as f:
                    f.write(input_json)
                mlflow.log_artifact(f"/tmp/prediction_{prediction_id}_input.json")
                
                # Log metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        mlflow.log_param(f"meta_{key}", str(value)[:250])  # MLflow param limit
                        
        except Exception as e:
            logger.error(f"Failed to log prediction to MLflow: {e}")
    
    def get_model_metrics(self, model_name: str, version: Optional[str] = None) -> Dict[str, float]:
        """Get metrics for a specific model version."""
        if not MLFLOW_AVAILABLE or not self.client:
            return {}
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
            else:
                versions = self.client.search_model_versions(
                    filter_string=f"name='{model_name}' and current_stage='{settings.MODEL_STAGE}'"
                )
                if versions:
                    model_version = versions[0]
                else:
                    return {}
            
            # Get run data
            run = self.client.get_run(model_version.run_id)
            metrics = run.data.metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get model metrics: {e}")
            return {}
    
    def get_model_artifacts(self, model_name: str, version: Optional[str] = None) -> List[str]:
        """Get list of artifacts for a model."""
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
            else:
                versions = self.client.search_model_versions(
                    filter_string=f"name='{model_name}' and current_stage='{settings.MODEL_STAGE}'"
                )
                if versions:
                    model_version = versions[0]
                else:
                    return []
            
            # List artifacts
            artifacts = self.client.list_artifacts(model_version.run_id)
            artifact_paths = [artifact.path for artifact in artifacts]
            
            return artifact_paths
            
        except Exception as e:
            logger.error(f"Failed to get model artifacts: {e}")
            return []
    
    def download_artifact(self, model_name: str, artifact_path: str, version: Optional[str] = None) -> str:
        """Download a specific artifact from a model run."""
        if not MLFLOW_AVAILABLE or not self.client:
            raise RuntimeError("MLflow is not available")
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
            else:
                versions = self.client.search_model_versions(
                    filter_string=f"name='{model_name}' and current_stage='{settings.MODEL_STAGE}'"
                )
                if versions:
                    model_version = versions[0]
                else:
                    raise ValueError(f"No model found for {model_name}")
            
            # Download artifact
            local_path = self.client.download_artifacts(model_version.run_id, artifact_path)
            
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download artifact: {e}")
            raise
    
    def search_experiments(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for experiments."""
        try:
            if query:
                experiments = self.client.search_experiments(filter_string=query)
            else:
                experiments = self.client.search_experiments()
            
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "tags": exp.tags
                }
                for exp in experiments
            ]
            
        except Exception as e:
            logger.error(f"Failed to search experiments: {e}")
            return []
    
    def search_runs(self, experiment_ids: List[str], filter_string: str = "", max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for runs within experiments."""
        try:
            runs = self.client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=max_results
            )
            
            return [
                {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                }
                for run in runs
            ]
            
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []
    
    def register_model(self, run_id: str, model_name: str, artifact_path: str = "model") -> ModelVersion:
        """Register a model from a run."""
        if not MLFLOW_AVAILABLE or not self.client:
            raise RuntimeError("MLflow is not available")
        try:
            # Register the model
            model_uri = f"runs:/{run_id}/{artifact_path}"
            mv = mlflow.register_model(model_uri, model_name)
            
            return ModelVersion(
                name=mv.name,
                version=mv.version,
                stage=mv.current_stage,
                run_id=mv.run_id
            )
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def transition_model_stage(self, model_name: str, version: str, stage: str) -> None:
        """Transition a model version to a new stage."""
        if not MLFLOW_AVAILABLE or not self.client:
            raise RuntimeError("MLflow is not available")
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            raise
    
    def get_model_lineage(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get the lineage information for a model."""
        if not MLFLOW_AVAILABLE or not self.client:
            return {}
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
            else:
                versions = self.client.search_model_versions(
                    filter_string=f"name='{model_name}' and current_stage='{settings.MODEL_STAGE}'"
                )
                if versions:
                    model_version = versions[0]
                else:
                    return {}
            
            # Get run information
            run = self.client.get_run(model_version.run_id)
            
            # Get experiment information
            experiment = self.client.get_experiment(run.info.experiment_id)
            
            return {
                "model_name": model_name,
                "version": model_version.version,
                "stage": model_version.current_stage,
                "run_id": model_version.run_id,
                "experiment_name": experiment.name,
                "created_at": model_version.creation_timestamp,
                "updated_at": model_version.last_updated_timestamp,
                "metrics": run.data.metrics,
                "parameters": run.data.params,
                "tags": {**run.data.tags, **model_version.tags},
                "artifacts": [a.path for a in self.client.list_artifacts(model_version.run_id)]
            }
            
        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            return {}


# Create singleton instance
mlflow_service = MLflowService()
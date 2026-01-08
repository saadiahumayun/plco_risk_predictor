#!/usr/bin/env python
"""
Deploy models from MLflow to production.

Usage:
    python scripts/deploy_models.py --ga-run-id <run_id> --baseline-run-id <run_id>
"""
import os
import sys
import argparse
import mlflow
from mlflow.tracking import MlflowClient
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def deploy_model(client: MlflowClient, 
                model_name: str, 
                run_id: str, 
                stage: str = "Production"):
    """Deploy a model to the specified stage."""
    try:
        # Register the model
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Registering model {model_name} from run {run_id}")
        
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered {model_name} version {model_version.version}")
        
        # Add description and tags
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"Deployed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Transition to stage
        logger.info(f"Transitioning {model_name} v{model_version.version} to {stage}")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage,
            archive_existing_versions=True
        )
        
        logger.info(f"Successfully deployed {model_name} to {stage}")
        return model_version.version
        
    except Exception as e:
        logger.error(f"Failed to deploy {model_name}: {e}")
        raise


def validate_model(client: MlflowClient, run_id: str) -> dict:
    """Validate model has required artifacts and metrics."""
    try:
        # Get run info
        run = client.get_run(run_id)
        
        # Check metrics
        metrics = run.data.metrics
        required_metrics = ["auc", "precision", "recall", "f1"]
        
        missing_metrics = [m for m in required_metrics if m not in metrics]
        if missing_metrics:
            logger.warning(f"Missing metrics: {missing_metrics}")
        
        # Check artifacts
        artifacts = client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]
        
        required_artifacts = ["model", "scaler", "features"]
        missing_artifacts = [
            a for a in required_artifacts 
            if not any(a in path for path in artifact_paths)
        ]
        
        if missing_artifacts:
            logger.warning(f"Missing artifacts: {missing_artifacts}")
        
        return {
            "metrics": metrics,
            "artifacts": artifact_paths,
            "missing_metrics": missing_metrics,
            "missing_artifacts": missing_artifacts
        }
        
    except Exception as e:
        logger.error(f"Failed to validate model: {e}")
        raise


def compare_models(client: MlflowClient, ga_run_id: str, baseline_run_id: str):
    """Compare GA and baseline model metrics."""
    try:
        ga_run = client.get_run(ga_run_id)
        baseline_run = client.get_run(baseline_run_id)
        
        ga_metrics = ga_run.data.metrics
        baseline_metrics = baseline_run.data.metrics
        
        logger.info("Model Comparison:")
        logger.info(f"{'Metric':<15} {'GA Model':<15} {'Baseline Model':<15} {'Improvement':<15}")
        logger.info("-" * 60)
        
        for metric in ["auc", "precision", "recall", "f1"]:
            ga_value = ga_metrics.get(metric, 0)
            baseline_value = baseline_metrics.get(metric, 0)
            improvement = ((ga_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0
            
            logger.info(
                f"{metric:<15} {ga_value:<15.4f} {baseline_value:<15.4f} "
                f"{improvement:+.2f}%"
            )
            
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")


def main():
    parser = argparse.ArgumentParser(description="Deploy models to production")
    parser.add_argument(
        "--ga-run-id", 
        required=True,
        help="MLflow run ID for GA-optimized model"
    )
    parser.add_argument(
        "--baseline-run-id",
        required=True,
        help="MLflow run ID for baseline model"
    )
    parser.add_argument(
        "--mlflow-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--stage",
        default="Production",
        choices=["Staging", "Production"],
        help="Deployment stage"
    )
    parser.add_argument(
        "--ga-model-name",
        default="breast_cancer_ga_model",
        help="Name for GA model in registry"
    )
    parser.add_argument(
        "--baseline-model-name",
        default="breast_cancer_baseline_model",
        help="Name for baseline model in registry"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate models, don't deploy"
    )
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient()
    
    logger.info(f"MLflow tracking URI: {args.mlflow_uri}")
    
    # Validate models
    logger.info("Validating GA model...")
    ga_validation = validate_model(client, args.ga_run_id)
    
    logger.info("Validating baseline model...")
    baseline_validation = validate_model(client, args.baseline_run_id)
    
    # Compare models
    compare_models(client, args.ga_run_id, args.baseline_run_id)
    
    if args.validate_only:
        logger.info("Validation complete (--validate-only flag set)")
        return
    
    # Check for critical missing items
    if ga_validation["missing_artifacts"] or baseline_validation["missing_artifacts"]:
        logger.error("Critical artifacts missing. Cannot deploy.")
        sys.exit(1)
    
    # Deploy models
    logger.info(f"Deploying models to {args.stage}...")
    
    ga_version = deploy_model(
        client,
        args.ga_model_name,
        args.ga_run_id,
        args.stage
    )
    
    baseline_version = deploy_model(
        client,
        args.baseline_model_name,
        args.baseline_run_id,
        args.stage
    )
    
    logger.info("\nDeployment Summary:")
    logger.info(f"GA Model: {args.ga_model_name} v{ga_version} -> {args.stage}")
    logger.info(f"Baseline Model: {args.baseline_model_name} v{baseline_version} -> {args.stage}")
    
    # Test deployment
    logger.info("\nTesting deployment...")
    try:
        ga_model = mlflow.pyfunc.load_model(f"models:/{args.ga_model_name}/{args.stage}")
        baseline_model = mlflow.pyfunc.load_model(f"models:/{args.baseline_model_name}/{args.stage}")
        logger.info("✓ Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load deployed models: {e}")
        sys.exit(1)
    
    logger.info("\nDeployment complete!")
    logger.info("Next steps:")
    logger.info("1. Restart the API service to load new models")
    logger.info("2. Run health checks: curl http://localhost:8000/api/v1/health/models")
    logger.info("3. Test predictions: curl -X POST http://localhost:8000/api/v1/predict")


if __name__ == "__main__":
    main()
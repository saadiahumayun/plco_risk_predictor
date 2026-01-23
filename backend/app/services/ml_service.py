# app/services/ml_service.py
"""
Machine Learning service for breast cancer risk prediction.
Integrates with MLflow for model loading and prediction tracking.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import time
import json
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from pathlib import Path

from app.core.config import settings, FEATURE_MAPPINGS, CLINICAL_FEATURES
from app.services.mlflow_service import mlflow_service
from app.models.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    RiskCategory,
    FeatureImportance,
    ModelComparison,
    ModelPrediction,
    ClinicalRecommendation,
    ScreeningRecommendation
)
from app.services.preprocessing import PreprocessingService
from app.utils.metrics import calculate_confidence_interval


logger = logging.getLogger(__name__)


class MLService:
    """Service for ML predictions and model management."""
    
    def __init__(self):
        self.ga_model = None
        self.baseline_model = None
        self.feature_scaler = None
        self.preprocessing_service = PreprocessingService()
        
        # Feature names
        self.ga_features = settings.GA_SELECTED_FEATURES
        self.baseline_features = []  # Will be loaded with model
        
        # Model metadata
        self.ga_model_version = None
        self.baseline_model_version = None
        
        # Performance tracking
        self.prediction_count = 0
        self.last_reload_time = None
        
        # Demo mode flag - set when MLflow models aren't available
        self.demo_mode = False
        
        # Load models on initialization
        self._load_models()
        
    def _load_models(self):
        """Load ML models - try pickle file first, then MLflow, then demo mode."""
        
        # Multiple possible locations for the model pickle file
        model_locations = [
            # Railway/Docker deployment path
            Path("/app/models/ga_model.pkl"),
            # Backend models directory
            Path(__file__).parent.parent.parent / "models" / "ga_model.pkl",
            # Local development path (your machine)
            Path("/Users/saadiahumayun/Documents/Thesis experiments/ga_f1_multi_population_experiment.pkl"),
            # Alternative names
            Path("/app/models/ga_f1_multi_population_experiment.pkl"),
        ]
        
        # Find the first existing model file
        pickle_path = None
        for path in model_locations:
            if path.exists():
                pickle_path = path
                logger.info(f"Found model at: {pickle_path}")
                break
        
        if pickle_path is None:
            logger.warning("No local model file found, will try MLflow...")
            pickle_path = Path("/nonexistent")  # Will fail exists() check
        
        # Try loading from pickle file first
        if pickle_path.exists():
            try:
                logger.info(f"Loading model from pickle file: {pickle_path}")
                # Try joblib first, then pickle
                try:
                    experiment_data = joblib.load(pickle_path)
                except (AttributeError, ModuleNotFoundError) as e:
                    logger.warning(f"Joblib failed (likely custom class issue): {e}")
                    import pickle
                    with open(pickle_path, 'rb') as f:
                        experiment_data = pickle.load(f)
                
                # Extract the best model from the experiment data
                if isinstance(experiment_data, dict):
                    # The pickle contains experiment results - extract best model
                    if 'best_model' in experiment_data:
                        self.ga_model = experiment_data['best_model']
                    elif 'models' in experiment_data and len(experiment_data['models']) > 0:
                        # Use the first/best model from the models list
                        self.ga_model = experiment_data['models'][0]
                    elif 'population_results' in experiment_data:
                        # Try to get model from population results
                        for pop_result in experiment_data.get('population_results', []):
                            if 'model' in pop_result:
                                self.ga_model = pop_result['model']
                                break
                    else:
                        # Log what keys are available for debugging
                        logger.info(f"Pickle file keys: {list(experiment_data.keys())}")
                        raise ValueError("Could not find model in pickle file structure")
                else:
                    # Direct model object
                    self.ga_model = experiment_data
                
                # Check if model has predict method
                if hasattr(self.ga_model, 'predict') or hasattr(self.ga_model, 'predict_proba'):
                    self.ga_model_version = "pickle-1.0"
                    self.baseline_model = self.ga_model  # Use same model as baseline
                    self.baseline_model_version = "pickle-1.0"
                    self.baseline_features = self.ga_features
                    self.feature_scaler = MinMaxScaler()
                    self.last_reload_time = datetime.utcnow()
                    logger.info("Successfully loaded model from pickle file!")
                    return
                else:
                    logger.warning("Loaded object doesn't have predict method")
                    
            except Exception as e:
                logger.warning(f"Failed to load from pickle file: {e}")
        
        # Try MLflow as fallback
        try:
            logger.info("Trying to load ML models from MLflow...")
            
            # Load GA model
            self.ga_model = mlflow_service.load_model(
                model_name=settings.GA_MODEL_NAME,
                stage=settings.MODEL_STAGE
            )
            self.ga_model_version = mlflow_service.ga_model_metadata.version if mlflow_service.ga_model_metadata else "unknown"
            logger.info(f"Loaded GA model version: {self.ga_model_version}")
            
            # Load baseline model
            self.baseline_model = mlflow_service.load_model(
                model_name=settings.BASELINE_MODEL_NAME,
                stage=settings.MODEL_STAGE
            )
            self.baseline_model_version = mlflow_service.baseline_model_metadata.version if mlflow_service.baseline_model_metadata else "unknown"
            logger.info(f"Loaded baseline model version: {self.baseline_model_version}")
            
            # Load feature scaler from MLflow artifacts
            self._load_feature_scaler()
            
            # Load baseline feature names from model metadata
            self._load_baseline_features()
            
            self.last_reload_time = datetime.utcnow()
            
        except Exception as e:
            logger.warning(f"Failed to load models from MLflow: {e}")
            logger.info("Running in DEMO MODE with mock predictions")
            self.demo_mode = True
            self.ga_model_version = "demo-1.0"
            self.baseline_model_version = "demo-1.0"
            self.baseline_features = self.ga_features
            self.feature_scaler = MinMaxScaler()
            self.last_reload_time = datetime.utcnow()
    
    def _load_feature_scaler(self):
        """Load feature scaler from MLflow artifacts."""
        try:
            # Try to download scaler artifact from GA model run
            scaler_path = mlflow_service.download_artifact(
                model_name=settings.GA_MODEL_NAME,
                artifact_path="scaler/feature_scaler.pkl"
            )
            
            self.feature_scaler = joblib.load(scaler_path)
            logger.info("Loaded feature scaler from MLflow")
            
        except Exception as e:
            logger.warning(f"Could not load scaler from MLflow, creating new one: {e}")
            self.feature_scaler = MinMaxScaler()
    
    def _load_baseline_features(self):
        """Load baseline model feature names from MLflow."""
        try:
            # Try to download feature names artifact
            features_path = mlflow_service.download_artifact(
                model_name=settings.BASELINE_MODEL_NAME,
                artifact_path="features/feature_names.json"
            )
            
            with open(features_path, 'r') as f:
                self.baseline_features = json.load(f)
            
            logger.info(f"Loaded {len(self.baseline_features)} baseline features")
            
        except Exception as e:
            logger.warning(f"Could not load baseline features from MLflow: {e}")
            # Use all features except target
            self.baseline_features = [f for f in settings.GA_SELECTED_FEATURES]
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make a risk prediction."""
        start_time = time.time()
        prediction_id = f"{datetime.utcnow().timestamp()}_{self.prediction_count}"
        self.prediction_count += 1
        
        try:
            # Preprocess input
            features_dict = self.preprocessing_service.preprocess_input(request)
            
            # Make predictions with both models
            ga_prediction = self._predict_with_ga_model(features_dict)
            baseline_prediction = self._predict_with_baseline_model(features_dict)
            
            # Use GA model as primary (based on your results showing better performance)
            primary_prediction = ga_prediction
            risk_score = float(primary_prediction)
            
            # Calculate confidence interval
            ci_lower, ci_upper = calculate_confidence_interval(risk_score)
            
            # Determine risk category
            risk_category = self._determine_risk_category(risk_score)
            
            # Get feature importance
            feature_importance = self._get_feature_importance(features_dict)
            
            # Calculate percentile (compared to population)
            percentile = self._calculate_risk_percentile(risk_score)
            
            # Calculate relative risk compared to average
            age = features_dict.get('age', 55)
            relative_risk = self._calculate_relative_risk(risk_score, age)
            
            # Model comparison
            model_comparison = ModelComparison(
                ga_model=ModelPrediction(
                    risk_score=float(ga_prediction),
                    features_used=len(self.ga_features),
                    model_version=self.ga_model_version
                ),
                baseline_model=ModelPrediction(
                    risk_score=float(baseline_prediction),
                    features_used=len(self.baseline_features),
                    model_version=self.baseline_model_version
                ),
                agreement=float(1 - abs(ga_prediction - baseline_prediction)),
                recommended_model="ga_model" if abs(ga_prediction - baseline_prediction) < 0.05 else "review_needed"
            )
            
            # Generate recommendations based on risk category
            recommendations = self._generate_recommendations(risk_category, features_dict)
            screening = self._generate_screening_recommendation(risk_category, features_dict)
            
            # Create response
            latency_ms = (time.time() - start_time) * 1000
            
            response = PredictionResponse(
                prediction_id=prediction_id,
                risk_score=risk_score,
                risk_category=risk_category,
                confidence_interval={"lower": ci_lower, "upper": ci_upper},
                percentile=percentile,
                relative_risk=relative_risk,
                feature_importance=feature_importance,
                model_comparison=model_comparison,
                recommendations=recommendations,
                screening=screening,
                model_version=self.ga_model_version,
                processing_time_ms=latency_ms
            )
            
            # Log prediction to MLflow
            if settings.LOG_PREDICTIONS:
                mlflow_service.log_prediction(
                    prediction_id=prediction_id,
                    model_name=settings.GA_MODEL_NAME,
                    input_features=features_dict,
                    prediction=risk_score,
                    risk_category=risk_category.value,
                    latency_ms=latency_ms,
                    metadata={
                        "baseline_prediction": float(baseline_prediction),
                        "model_agreement": model_comparison.agreement,
                        "patient_age": request.demographics.age
                    }
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _predict_with_ga_model(self, features: Dict[str, Any]) -> float:
        """Make prediction with GA-selected features model."""
        # Demo mode: return mock prediction based on risk factors
        if self.demo_mode:
            return self._generate_mock_prediction(features)
        
        # Create feature vector in correct order
        feature_vector = []
        for feature_name in self.ga_features:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                # Handle missing features with defaults
                logger.warning(f"Missing feature: {feature_name}, using default")
                feature_vector.append(0)
        
        # Convert to DataFrame for prediction
        X = pd.DataFrame([feature_vector], columns=self.ga_features)
        
        # Scale features if scaler is available
        if self.feature_scaler is not None:
            try:
                X_scaled = self.feature_scaler.transform(X)
                X = pd.DataFrame(X_scaled, columns=self.ga_features)
            except Exception as e:
                logger.warning(f"Could not scale features: {e}")
        
        # Make prediction
        pred = self.ga_model.predict_proba(X)[0, 1]  # Probability of positive class
        return float(pred)
    
    def _generate_mock_prediction(self, features: Dict[str, Any]) -> float:
        """Generate a realistic mock prediction based on input features."""
        import random
        
        # Base risk (average population risk ~12% for 5-year)
        base_risk = 0.12
        
        # Adjust based on age (if available)
        age = features.get('age', 50)
        if age > 60:
            base_risk += 0.03
        elif age > 50:
            base_risk += 0.015
        
        # Adjust based on family history
        family_history = features.get('first_degree_bc', features.get('number_of_relatives_with_bc', 0))
        if family_history and family_history > 0:
            base_risk += 0.045 * family_history
        
        # Adjust based on BMI
        bmi = features.get('current_bmi', 25)
        if bmi > 30:
            base_risk += 0.02
        elif bmi > 25:
            base_risk += 0.01
        
        # Add some randomness
        base_risk += random.uniform(-0.02, 0.02)
        
        # Clamp to valid range
        return max(0.01, min(0.95, base_risk))
    
    def _predict_with_baseline_model(self, features: Dict[str, Any]) -> float:
        """Make prediction with baseline model (all features)."""
        # Demo mode: return slightly different mock prediction
        if self.demo_mode:
            ga_pred = self._generate_mock_prediction(features)
            # Baseline model gives slightly different result
            return ga_pred + np.random.uniform(-0.01, 0.01)
        
        # Create feature vector
        feature_vector = []
        feature_names = self.baseline_features if self.baseline_features else self.ga_features
        
        for feature_name in feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0)
        
        # Convert to DataFrame
        X = pd.DataFrame([feature_vector], columns=feature_names)
        
        # Scale if available
        if self.feature_scaler is not None:
            try:
                X_scaled = self.feature_scaler.transform(X)
                X = pd.DataFrame(X_scaled, columns=feature_names)
            except Exception as e:
                logger.warning(f"Could not scale baseline features: {e}")
        
        # Make prediction
        pred = self.baseline_model.predict_proba(X)[0, 1]
        return float(pred)
    
    def _determine_risk_category(self, risk_score: float) -> RiskCategory:
        """Determine risk category based on score."""
        if risk_score >= settings.HIGH_RISK_THRESHOLD:
            return RiskCategory.HIGH
        elif risk_score >= settings.MODERATE_RISK_THRESHOLD:
            return RiskCategory.MODERATE
        else:
            return RiskCategory.LOW
    
    def _get_feature_importance(self, features: Dict[str, Any]) -> List[FeatureImportance]:
        """Get feature importance based on user inputs and clinical knowledge."""
        feature_importance_list = []
        
        # Age - major risk factor
        age = features.get('age', 50)
        if age >= 60:
            feature_importance_list.append(FeatureImportance(
                feature='age', importance=0.15, value=float(age),
                contribution=0.04, description=f'Age {int(age)} years - higher risk with advancing age'
            ))
        elif age >= 50:
            feature_importance_list.append(FeatureImportance(
                feature='age', importance=0.12, value=float(age),
                contribution=0.02, description=f'Age {int(age)} years - moderate age-related risk'
            ))
        else:
            feature_importance_list.append(FeatureImportance(
                feature='age', importance=0.08, value=float(age),
                contribution=-0.01, description=f'Age {int(age)} years - younger age is protective'
            ))
        
        # Family history
        fh_cancer = features.get('fh_cancer', 0)
        if fh_cancer > 0:
            feature_importance_list.append(FeatureImportance(
                feature='family_history', importance=0.18, value=float(fh_cancer),
                contribution=0.05, description='Family history of breast cancer increases risk'
            ))
        else:
            feature_importance_list.append(FeatureImportance(
                feature='family_history', importance=0.10, value=0.0,
                contribution=-0.02, description='No family history of breast cancer'
            ))
        
        # BMI
        bmi = features.get('bmi_curr', 25)
        if bmi >= 30:
            feature_importance_list.append(FeatureImportance(
                feature='bmi', importance=0.10, value=float(bmi),
                contribution=0.03, description=f'BMI {bmi:.1f} - obesity increases postmenopausal risk'
            ))
        elif bmi >= 25:
            feature_importance_list.append(FeatureImportance(
                feature='bmi', importance=0.08, value=float(bmi),
                contribution=0.01, description=f'BMI {bmi:.1f} - slightly elevated BMI'
            ))
        else:
            feature_importance_list.append(FeatureImportance(
                feature='bmi', importance=0.06, value=float(bmi),
                contribution=-0.01, description=f'BMI {bmi:.1f} - healthy weight is protective'
            ))
        
        # Hormone therapy
        horm_f = features.get('horm_f', 0)
        if horm_f > 0:
            feature_importance_list.append(FeatureImportance(
                feature='hormone_therapy', importance=0.12, value=1.0,
                contribution=0.03, description='Hormone therapy use increases risk'
            ))
        
        # Smoking
        smoked_f = features.get('smoked_f', 0)
        if smoked_f > 0:
            feature_importance_list.append(FeatureImportance(
                feature='smoking', importance=0.08, value=1.0,
                contribution=0.02, description='Smoking history associated with increased risk'
            ))
        else:
            feature_importance_list.append(FeatureImportance(
                feature='smoking', importance=0.05, value=0.0,
                contribution=-0.01, description='Non-smoker - reduced risk factor'
            ))
        
        # Age at menarche
        fmenstr = features.get('fmenstr', 3)
        if fmenstr <= 2:  # Early menarche
            feature_importance_list.append(FeatureImportance(
                feature='age_at_menarche', importance=0.06, value=float(fmenstr),
                contribution=0.015, description='Early age at first menstruation'
            ))
        elif fmenstr >= 4:  # Late menarche
            feature_importance_list.append(FeatureImportance(
                feature='age_at_menarche', importance=0.06, value=float(fmenstr),
                contribution=-0.01, description='Later age at first menstruation is protective'
            ))
        
        # Live births (parity)
        livec = features.get('livec', 0)
        if livec >= 2:
            feature_importance_list.append(FeatureImportance(
                feature='parity', importance=0.07, value=float(livec),
                contribution=-0.02, description=f'{int(livec)} live births - childbearing is protective'
            ))
        elif livec == 0:
            feature_importance_list.append(FeatureImportance(
                feature='parity', importance=0.07, value=0.0,
                contribution=0.015, description='Nulliparity associated with slightly higher risk'
            ))
        
        # Birth control
        bcontr_f = features.get('bcontr_f', 0)
        if bcontr_f == 0:
            feature_importance_list.append(FeatureImportance(
                feature='birth_control', importance=0.04, value=0.0,
                contribution=-0.005, description='No oral contraceptive use'
            ))
        
        # Benign breast disease (from preprocessed features or request)
        bbd = features.get('bbd', 0)
        if bbd > 0:
            feature_importance_list.append(FeatureImportance(
                feature='benign_breast_disease', importance=0.10, value=1.0,
                contribution=0.025, description='History of benign breast disease increases risk'
            ))
        else:
            feature_importance_list.append(FeatureImportance(
                feature='benign_breast_disease', importance=0.05, value=0.0,
                contribution=-0.01, description='No history of benign breast disease'
            ))
                
                # Sort by absolute contribution
                feature_importance_list.sort(key=lambda x: abs(x.contribution), reverse=True)
                
        return feature_importance_list[:8]  # Top 8 factors
    
    def _calculate_risk_percentile(self, risk_score: float) -> int:
        """Calculate risk percentile compared to population."""
        # Based on typical breast cancer risk distributions
        # These would ideally come from your actual population data
        population_distribution = [
            (0.01, 10), (0.02, 20), (0.03, 30), (0.05, 40),
            (0.08, 50), (0.12, 60), (0.16, 70), (0.20, 80),
            (0.25, 90), (0.30, 95), (0.40, 99)
        ]
        
        for threshold, percentile in population_distribution:
            if risk_score <= threshold:
                return percentile
        
        return 99  # Top 1%
    
    def _calculate_relative_risk(self, risk_score: float, age: int) -> float:
        """Calculate relative risk compared to average women of the same age.
        
        These are illustrative average 5-year risks by age group from epidemiological data.
        """
        # Average 5-year breast cancer risk by age group
        # Based on general population statistics
        average_risk_by_age = {
            (40, 50): 0.010,  # 1.0%
            (50, 60): 0.015,  # 1.5%
            (60, 70): 0.020,  # 2.0%
            (70, 80): 0.025,  # 2.5%
        }
        
        # Find the average risk for this age group
        avg_risk = 0.015  # Default average risk (1.5%)
        for (low, high), risk in average_risk_by_age.items():
            if low <= age < high:
                avg_risk = risk
                break
        
        # Calculate relative risk
        if avg_risk > 0:
            return round(risk_score / avg_risk, 2)
        return 1.0
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description for a feature."""
        descriptions = {
            "age": "Current age",
            "bmi_curr": "Current body mass index",
            "fmenstr": "Age at first menstruation",
            "fchilda": "Age at first live birth",
            "race7": "Race/ethnicity",
            "fh_cancer": "Family history of cancer",
            "ph_any_trial": "Personal history of cancer",
            "livec": "Number of live births",
            "menstrs": "Menopause status",
            "ibup": "Ibuprofen use",
            # Add more descriptions
        }
        return descriptions.get(feature_name, feature_name.replace("_", " ").title())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "ga_model": {
                "name": settings.GA_MODEL_NAME,
                "version": self.ga_model_version,
                "features": len(self.ga_features),
                "stage": settings.MODEL_STAGE,
                "loaded": self.ga_model is not None
            },
            "baseline_model": {
                "name": settings.BASELINE_MODEL_NAME,
                "version": self.baseline_model_version,
                "features": len(self.baseline_features),
                "stage": settings.MODEL_STAGE,
                "loaded": self.baseline_model is not None
            },
            "last_reload": self.last_reload_time.isoformat() if self.last_reload_time else None,
            "prediction_count": self.prediction_count,
            "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI
        }
    
    def reload_models(self):
        """Reload models from MLflow."""
        logger.info("Reloading models from MLflow...")
        self._load_models()
        return {"status": "success", "message": "Models reloaded successfully"}
    
    def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """Make batch predictions."""
        responses = []
        for request in requests:
            try:
                response = self.predict(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch prediction failed for one request: {e}")
                # Continue with other predictions
        
        return responses
    
    def _generate_recommendations(self, risk_category: RiskCategory, features: Dict[str, Any]) -> List[ClinicalRecommendation]:
        """Generate clinical recommendations based on risk category and features."""
        recommendations = []
        
        # Check family history for genetic counseling
        fh_cancer = features.get('fh_cancer', 0)
        
        if risk_category == RiskCategory.HIGH:
            # High risk recommendations
            recommendations.append(ClinicalRecommendation(
                type="screening",
                priority="high",
                action="Enhanced screening with annual mammography and consider breast MRI",
                rationale="High risk category warrants more intensive surveillance"
            ))
            recommendations.append(ClinicalRecommendation(
                type="medical",
                priority="high",
                action="Discuss chemoprevention options (tamoxifen/raloxifene)",
                rationale=f"5-year risk exceeds {settings.HIGH_RISK_THRESHOLD*100:.1f}% threshold for chemoprevention"
            ))
            if fh_cancer > 0:
                recommendations.append(ClinicalRecommendation(
                    type="genetic",
                    priority="high",
                    action="Genetic counseling and BRCA testing recommended",
                    rationale="Family history combined with high risk score"
                ))
        elif risk_category == RiskCategory.MODERATE:
            # Moderate risk recommendations
            recommendations.append(ClinicalRecommendation(
                type="screening",
                priority="medium",
                action="Annual mammography recommended",
                rationale="Moderate risk warrants regular surveillance"
            ))
            if fh_cancer > 0:
                recommendations.append(ClinicalRecommendation(
                    type="genetic",
                    priority="medium",
                    action="Consider genetic counseling",
                    rationale="Family history present"
                ))
            recommendations.append(ClinicalRecommendation(
                type="lifestyle",
                priority="medium",
                action="Lifestyle modifications to reduce risk",
                rationale="Weight management, physical activity, and limiting alcohol"
            ))
        else:
            # Low risk recommendations
            recommendations.append(ClinicalRecommendation(
                type="screening",
                priority="low",
                action="Standard screening per age-appropriate guidelines",
                rationale="Low risk - follow standard screening protocols"
            ))
            recommendations.append(ClinicalRecommendation(
                type="lifestyle",
                priority="low",
                action="Maintain healthy lifestyle",
                rationale="Continue healthy habits for risk maintenance"
            ))
        
        return recommendations
    
    def _generate_screening_recommendation(self, risk_category: RiskCategory, features: Dict[str, Any]) -> ScreeningRecommendation:
        """Generate screening recommendations based on risk category."""
        from datetime import datetime, timedelta
        
        age = features.get('age', 50)
        fh_cancer = features.get('fh_cancer', 0)
        bbd = features.get('bbd', 0)
        
        if risk_category == RiskCategory.HIGH:
            additional = ["MRI", "Ultrasound"]
            return ScreeningRecommendation(
                recommendation="Enhanced Screening Protocol",
                frequency="Every 6 months",
                rationale="High risk category requires more intensive surveillance including breast MRI",
                next_date=datetime.utcnow() + timedelta(days=180),
                additional_imaging=additional
            )
        elif risk_category == RiskCategory.MODERATE:
            additional = []
            if fh_cancer > 1:
                additional.append("MRI")
            if bbd > 0 or age < 50:
                additional.append("Ultrasound")
            return ScreeningRecommendation(
                recommendation="Annual Mammography",
                frequency="Annually",
                rationale="Moderate risk warrants annual screening mammography",
                next_date=datetime.utcnow() + timedelta(days=365),
                additional_imaging=additional if additional else None
            )
        else:
            return ScreeningRecommendation(
                recommendation="Standard Screening Guidelines",
                frequency="Every 1-2 years per guidelines",
                rationale="Low risk - follow age-appropriate screening guidelines",
                next_date=datetime.utcnow() + timedelta(days=365 if age >= 50 else 730),
                additional_imaging=None
            )


# Create singleton instance
ml_service = MLService()
# tests/test_api.py
"""
API endpoint tests for breast cancer risk prediction.
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

from app.main import app
from app.models.schemas import PredictionRequest, Demographics, ReproductiveHistory, BodyMetrics, MedicalHistory


client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "name" in response.json()
        assert "version" in response.json()
    
    def test_health_check(self):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_detailed_health(self):
        """Test detailed health check."""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert "api" in data
        assert "models" in data
        assert "database" in data


class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    @pytest.fixture
    def valid_prediction_request(self):
        """Create a valid prediction request."""
        return {
            "demographics": {
                "age": 55,
                "race": "white",
                "education": "college_graduate",
                "marital_status": "married"
            },
            "reproductive_history": {
                "age_at_menarche": 12,
                "age_at_first_birth": 28,
                "number_of_live_births": 2,
                "first_degree_bc": 1,
                "breastfeeding_months": 12
            },
            "body_metrics": {
                "current_bmi": 26.5,
                "bmi_at_age_20": 22.0,
                "height_cm": 165,
                "weight_kg": 72
            },
            "medical_history": {
                "personal_cancer_history": False,
                "benign_breast_disease": True,
                "breast_biopsies": 1,
                "hormone_therapy_current": False,
                "hormone_therapy_years": 0,
                "aspirin_use": True,
                "ibuprofen_use": False
            }
        }
    
    def test_valid_prediction(self, valid_prediction_request):
        """Test valid prediction request."""
        response = client.post("/api/v1/predict", json=valid_prediction_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction_id" in data
        assert "risk_score" in data
        assert 0 <= data["risk_score"] <= 1
        assert data["risk_category"] in ["low", "moderate", "high"]
        assert "confidence_interval" in data
        assert "percentile" in data
        assert "feature_importance" in data
    
    def test_invalid_age(self, valid_prediction_request):
        """Test prediction with invalid age."""
        valid_prediction_request["demographics"]["age"] = 150
        response = client.post("/api/v1/predict", json=valid_prediction_request)
        assert response.status_code == 422
        assert "Age must be between" in str(response.json())
    
    def test_invalid_bmi(self, valid_prediction_request):
        """Test prediction with mismatched BMI."""
        valid_prediction_request["body_metrics"]["current_bmi"] = 50  # Doesn't match height/weight
        response = client.post("/api/v1/predict", json=valid_prediction_request)
        assert response.status_code == 422
        assert "BMI doesn't match" in str(response.json())
    
    def test_missing_required_field(self, valid_prediction_request):
        """Test prediction with missing required field."""
        del valid_prediction_request["demographics"]["age"]
        response = client.post("/api/v1/predict", json=valid_prediction_request)
        assert response.status_code == 422
    
    def test_batch_prediction(self, valid_prediction_request):
        """Test batch prediction endpoint."""
        batch_request = {
            "requests": [valid_prediction_request] * 3
        }
        response = client.post("/api/v1/predict/batch", json=batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        assert data["successful_predictions"] == 3
        assert data["total_requests"] == 3


class TestModelEndpoints:
    """Test model management endpoints."""
    
    def test_model_info(self):
        """Test model information endpoint."""
        response = client.get("/api/v1/models/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "ga_model" in data
        assert "baseline_model" in data
    
    def test_get_features(self):
        """Test feature information endpoint."""
        response = client.get("/api/v1/features")
        assert response.status_code == 200
        
        data = response.json()
        assert "ga_features" in data
        assert "baseline_features" in data
        assert data["ga_features"]["count"] == 28


class TestAnalyticsEndpoints:
    """Test analytics endpoints."""
    
    def test_prediction_stats(self):
        """Test prediction statistics endpoint."""
        response = client.get("/api/v1/analytics/predictions?time_period=24h")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_predictions" in data
        assert "average_risk_score" in data
        assert "risk_distribution" in data
    
    def test_model_performance(self):
        """Test model performance endpoint."""
        response = client.get("/api/v1/analytics/performance?days=7")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        if data:  # If there's performance data
            assert "model_name" in data[0]
            assert "auc" in data[0]
            assert "precision" in data[0]
            assert "recall" in data[0]
    
    def test_drift_analysis(self):
        """Test drift analysis endpoint."""
        response = client.get("/api/v1/analytics/drift?days=7")
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis_date" in data
        assert "features_analyzed" in data
        assert "drift_detected" in data


class TestValidationEndpoint:
    """Test input validation endpoint."""
    
    def test_valid_input(self):
        """Test validation with valid input."""
        request = {
            "demographics": {"age": 50, "race": "white"},
            "reproductive_history": {
                "age_at_menarche": 13,
                "number_of_live_births": 2,
                "age_at_first_birth": 25,
                "first_degree_bc": 0
            },
            "body_metrics": {
                "current_bmi": 25,
                "height_cm": 165,
                "weight_kg": 68
            },
            "medical_history": {
                "personal_cancer_history": False,
                "benign_breast_disease": False,
                "breast_biopsies": 0
            }
        }
        
        response = client.post("/api/v1/validate", json=request)
        assert response.status_code == 200
        assert response.json()["valid"] is True
        assert response.json()["errors"] == []
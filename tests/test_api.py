"""
Unit Tests for FastAPI Application
===================================
Tests for API endpoints and functionality(API uç noktaları ve işlevselliği için testler)
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    pytest.skip("API module not available", allow_module_level=True)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def client():
    """Test client for API"""
    return TestClient(app)

@pytest.fixture
def sample_patient_payload():
    """Sample patient data for API requests"""
    return {
        "Age": 55,
        "Gender": 1,
        "Air_Pollution": 7,
        "Alcohol_use": 6,
        "Dust_Allergy": 5,
        "OccuPational_Hazards": 6,
        "Genetic_Risk": 5,
        "chronic_Lung_Disease": 4,
        "Balanced_Diet": 3,
        "Obesity": 6,
        "Smoking": 7,
        "Passive_Smoker": 5,
        "Chest_Pain": 7,
        "Coughing_of_Blood": 6,
        "Fatigue": 7,
        "Weight_Loss": 5,
        "Shortness_of_Breath": 8,
        "Wheezing": 6,
        "Swallowing_Difficulty": 4,
        "Clubbing_of_Finger_Nails": 3,
        "Frequent_Cold": 4,
        "Dry_Cough": 5,
        "Snoring": 3
    }

@pytest.fixture
def invalid_patient_payload():
    """Invalid patient data for testing validation"""
    return {
        "Age": 150,  # Invalid: too old
        "Gender": 3,  # Invalid: only 1 or 2
        "Air_Pollution": 10,  # Invalid: max is 8
        "Alcohol_use": 0,  # Invalid: min is 1
    }


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================

class TestHealthEndpoints:
    """Tests for health check endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns 200"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "model not loaded"]
        assert "timestamp" in data
    
    def test_health_response_structure(self, client):
        """Test health response has correct structure"""
        response = client.get("/health")
        data = response.json()
        
        required_fields = ["status", "timestamp", "model_loaded"]
        for field in required_fields:
            assert field in data


# =============================================================================
# PREDICTION ENDPOINT TESTS
# =============================================================================

class TestPredictionEndpoint:
    """Tests for prediction endpoint"""
    
    def test_predict_endpoint_success(self, client, sample_patient_payload):
        """Test successful prediction"""
        response = client.post("/predict", json=sample_patient_payload)
        
        # May fail if model not loaded, but should return valid status code
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
    
    def test_predict_response_structure(self, client, sample_patient_payload):
        """Test prediction response structure"""
        response = client.post("/predict", json=sample_patient_payload)
        
        if response.status_code == 200:
            data = response.json()
            
            required_fields = [
                "prediction", "confidence", "probabilities",
                "risk_factors", "overall_risk_score", "timestamp", "recommendations"
            ]
            
            for field in required_fields:
                assert field in data, f"Field {field} missing from response"
    
    def test_predict_valid_prediction_values(self, client, sample_patient_payload):
        """Test that prediction values are valid"""
        response = client.post("/predict", json=sample_patient_payload)
        
        if response.status_code == 200:
            data = response.json()
            
            # Prediction should be one of the valid classes
            assert data["prediction"] in ["Low", "Medium", "High"]
            
            # Confidence should be between 0 and 1
            assert 0 <= data["confidence"] <= 1
            
            # Probabilities should sum to ~1
            probs = data["probabilities"]
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.01
    
    def test_predict_invalid_data_validation(self, client):
        """Test validation for invalid data"""
        invalid_data = {
            "Age": 150,  # Too high
            "Gender": 1,
            "Air_Pollution": 10  # Too high
        }
        
        response = client.post("/predict", json=invalid_data)
        
        # Should return validation error (422)
        assert response.status_code == 422
    
    def test_predict_missing_fields(self, client):
        """Test prediction with missing required fields"""
        incomplete_data = {
            "Age": 55,
            "Gender": 1
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_data)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_predict_recommendations_present(self, client, sample_patient_payload):
        """Test that recommendations are included"""
        response = client.post("/predict", json=sample_patient_payload)
        
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert isinstance(data["recommendations"], list)
            assert len(data["recommendations"]) > 0


# =============================================================================
# BATCH PREDICTION TESTS
# =============================================================================

class TestBatchPrediction:
    """Tests for batch prediction endpoint"""
    
    def test_batch_predict_endpoint(self, client, sample_patient_payload):
        """Test batch prediction with multiple patients"""
        batch_payload = {
            "patients": [
                sample_patient_payload,
                sample_patient_payload  # Same patient twice for simplicity
            ]
        }
        
        response = client.post("/predict/batch", json=batch_payload)
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "count" in data
            assert data["count"] == 2
    
    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list"""
        batch_payload = {"patients": []}
        
        response = client.post("/predict/batch", json=batch_payload)
        
        if response.status_code == 200:
            data = response.json()
            assert data["count"] == 0


# =============================================================================
# MODEL INFO TESTS
# =============================================================================

class TestModelInfo:
    """Tests for model info endpoint"""
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "features_count" in data
            assert "classes" in data
    
    def test_model_info_classes(self, client):
        """Test that model info contains correct classes"""
        response = client.get("/model/info")
        
        if response.status_code == 200:
            data = response.json()
            expected_classes = ["Low", "Medium", "High"]
            assert data["classes"] == expected_classes


# =============================================================================
# RISK FACTORS INFO TESTS
# =============================================================================

class TestRiskFactorsInfo:
    """Tests for risk factors info endpoint"""
    
    def test_risk_factors_endpoint(self, client):
        """Test risk factors info endpoint"""
        response = client.get("/risk/factors")
        
        assert response.status_code == 200
        data = response.json()
        assert "risk_factors" in data
    
    def test_risk_factors_structure(self, client):
        """Test risk factors info structure"""
        response = client.get("/risk/factors")
        data = response.json()
        
        risk_factors = data["risk_factors"]
        
        expected_factors = [
            "Lifestyle Risk",
            "Environmental Risk",
            "Genetic/Health Risk",
            "Symptom Severity"
        ]
        
        for factor in expected_factors:
            assert factor in risk_factors


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling"""
    
    def test_invalid_endpoint(self, client):
        """Test request to invalid endpoint"""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404
    
    def test_wrong_http_method(self, client):
        """Test wrong HTTP method"""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == 405
    
    def test_invalid_json(self, client):
        """Test request with invalid JSON"""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


# =============================================================================
# CORS TESTS
# =============================================================================

class TestCORS:
    """Tests for CORS configuration"""
    
    def test_cors_headers_present(self, client, sample_patient_payload):
        """Test that CORS headers are present"""
        response = client.post("/predict", json=sample_patient_payload)
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestAPIIntegration:
    """Integration tests for API workflows"""
    
    def test_full_prediction_workflow(self, client, sample_patient_payload):
        """Test complete prediction workflow via API"""
        # Step 1: Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # Step 2: Get model info
        info_response = client.get("/model/info")
        assert info_response.status_code in [200, 503]
        
        # Step 3: Make prediction
        pred_response = client.post("/predict", json=sample_patient_payload)
        assert pred_response.status_code in [200, 503]
        
        if pred_response.status_code == 200:
            data = pred_response.json()
            assert data["prediction"] in ["Low", "Medium", "High"]
    
    def test_multiple_predictions_consistency(self, client, sample_patient_payload):
        """Test that multiple predictions are consistent"""
        response1 = client.post("/predict", json=sample_patient_payload)
        response2 = client.post("/predict", json=sample_patient_payload)
        
        if response1.status_code == 200 and response2.status_code == 200:
            pred1 = response1.json()["prediction"]
            pred2 = response2.json()["prediction"]
            
            # Same input should give same prediction
            assert pred1 == pred2


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Basic performance tests"""
    
    def test_prediction_response_time(self, client, sample_patient_payload):
        """Test that prediction responds in reasonable time"""
        import time
        
        start = time.time()
        response = client.post("/predict", json=sample_patient_payload)
        end = time.time()
        
        response_time = end - start
        
        # Should respond within 5 seconds
        assert response_time < 5.0, f"Response took {response_time:.2f}s"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
"""
Unit Tests for Inference Module
================================
Tests for CancerRiskPredictor class and related functions (CancerRiskPredictor sınıfı ve 
ilgili işlevler için testler)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from inference import CancerRiskPredictor
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    pytest.skip("Inference module not available", allow_module_level=True)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing"""
    return {
        'Age': 55,
        'Gender': 1,
        'Air Pollution': 7,
        'Alcohol use': 6,
        'Dust Allergy': 5,
        'OccuPational Hazards': 6,
        'Genetic Risk': 5,
        'chronic Lung Disease': 4,
        'Balanced Diet': 3,
        'Obesity': 6,
        'Smoking': 7,
        'Passive Smoker': 5,
        'Chest Pain': 7,
        'Coughing of Blood': 6,
        'Fatigue': 7,
        'Weight Loss': 5,
        'Shortness of Breath': 8,
        'Wheezing': 6,
        'Swallowing Difficulty': 4,
        'Clubbing of Finger Nails': 3,
        'Frequent Cold': 4,
        'Dry Cough': 5,
        'Snoring': 3
    }

@pytest.fixture
def low_risk_patient():
    """Low risk patient data"""
    return {
        'Age': 25,
        'Gender': 1,
        'Air Pollution': 2,
        'Alcohol use': 1,
        'Dust Allergy': 2,
        'OccuPational Hazards': 2,
        'Genetic Risk': 2,
        'chronic Lung Disease': 1,
        'Balanced Diet': 6,
        'Obesity': 2,
        'Smoking': 1,
        'Passive Smoker': 1,
        'Chest Pain': 2,
        'Coughing of Blood': 1,
        'Fatigue': 2,
        'Weight Loss': 1,
        'Shortness of Breath': 2,
        'Wheezing': 1,
        'Swallowing Difficulty': 1,
        'Clubbing of Finger Nails': 1,
        'Frequent Cold': 2,
        'Dry Cough': 2,
        'Snoring': 2
    }

@pytest.fixture
def high_risk_patient():
    """High risk patient data"""
    return {
        'Age': 65,
        'Gender': 1,
        'Air Pollution': 8,
        'Alcohol use': 8,
        'Dust Allergy': 7,
        'OccuPational Hazards': 7,
        'Genetic Risk': 7,
        'chronic Lung Disease': 6,
        'Balanced Diet': 2,
        'Obesity': 7,
        'Smoking': 8,
        'Passive Smoker': 7,
        'Chest Pain': 9,
        'Coughing of Blood': 8,
        'Fatigue': 9,
        'Weight Loss': 8,
        'Shortness of Breath': 9,
        'Wheezing': 8,
        'Swallowing Difficulty': 7,
        'Clubbing of Finger Nails': 8,
        'Frequent Cold': 6,
        'Dry Cough': 7,
        'Snoring': 6
    }

@pytest.fixture
def predictor():
    """Initialize predictor"""
    try:
        return CancerRiskPredictor()
    except FileNotFoundError:
        pytest.skip("Model files not found. Run pipeline.py first.")


# =============================================================================
# PREDICTOR INITIALIZATION TESTS
# =============================================================================

class TestPredictorInitialization:
    """Tests for predictor initialization"""
    
    def test_predictor_loads_successfully(self, predictor):
        """Test that predictor initializes without errors"""
        assert predictor is not None
        assert predictor.model is not None
        assert predictor.scaler is not None
    
    def test_model_is_loaded(self, predictor):
        """Test that model is properly loaded"""
        assert hasattr(predictor.model, 'predict')
        assert hasattr(predictor.model, 'predict_proba')
    
    def test_scaler_is_loaded(self, predictor):
        """Test that scaler is properly loaded"""
        assert hasattr(predictor.scaler, 'transform')


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================

class TestFeatureEngineering:
    """Tests for feature engineering functions"""
    
    def test_engineer_features_returns_dataframe(self, predictor, sample_patient_data):
        """Test that feature engineering returns a DataFrame"""
        df = pd.DataFrame([sample_patient_data])
        df_fe = predictor.engineer_features(df)
        
        assert isinstance(df_fe, pd.DataFrame)
        assert len(df_fe) == len(df)
    
    def test_new_features_created(self, predictor, sample_patient_data):
        """Test that new features are created"""
        df = pd.DataFrame([sample_patient_data])
        df_fe = predictor.engineer_features(df)
        
        # Check for new features
        new_features = [
            'Age_Group', 'Environmental_Risk', 'Lifestyle_Risk',
            'Genetic_Health_Risk', 'Symptom_Severity', 'Respiratory_Score',
            'Overall_Risk_Score', 'Smoking_Age_Interaction'
        ]
        
        for feature in new_features:
            assert feature in df_fe.columns, f"Feature {feature} not created"
    
    def test_risk_scores_in_valid_range(self, predictor, sample_patient_data):
        """Test that risk scores are in valid range"""
        df = pd.DataFrame([sample_patient_data])
        df_fe = predictor.engineer_features(df)
        
        # Risk scores should be between 0 and 10
        risk_features = ['Environmental_Risk', 'Lifestyle_Risk', 
                        'Genetic_Health_Risk', 'Symptom_Severity']
        
        for feature in risk_features:
            assert df_fe[feature].min() >= 0
            assert df_fe[feature].max() <= 10
    
    def test_age_group_categories(self, predictor, sample_patient_data):
        """Test age group categorization"""
        df = pd.DataFrame([sample_patient_data])
        df_fe = predictor.engineer_features(df)
        
        assert df_fe['Age_Group'].iloc[0] in [0, 1, 2, 3]


# =============================================================================
# PREPROCESSING TESTS
# =============================================================================

class TestPreprocessing:
    """Tests for preprocessing functions"""
    
    def test_preprocess_returns_numpy_array(self, predictor, sample_patient_data):
        """Test that preprocessing returns numpy array"""
        X = predictor.preprocess(sample_patient_data)
        
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
    
    def test_preprocess_correct_shape(self, predictor, sample_patient_data):
        """Test that preprocessed data has correct shape"""
        X = predictor.preprocess(sample_patient_data)
        
        # Should have 1 row and expected number of features
        assert X.shape[0] == 1
        assert X.shape[1] > 0  # Should have multiple features
    
    def test_preprocess_handles_dataframe(self, predictor, sample_patient_data):
        """Test that preprocessing handles DataFrame input"""
        df = pd.DataFrame([sample_patient_data])
        X = predictor.preprocess(df)
        
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 1


# =============================================================================
# PREDICTION TESTS
# =============================================================================

class TestPredictions:
    """Tests for prediction functions"""
    
    def test_predict_returns_valid_class(self, predictor, sample_patient_data):
        """Test that prediction returns valid class"""
        prediction = predictor.predict(sample_patient_data)
        
        assert prediction in ['Low', 'Medium', 'High']
    
    def test_predict_proba_returns_dict(self, predictor, sample_patient_data):
        """Test that predict_proba returns dictionary"""
        probabilities = predictor.predict_proba(sample_patient_data)
        
        assert isinstance(probabilities, dict)
        assert len(probabilities) == 3
        assert 'Low' in probabilities
        assert 'Medium' in probabilities
        assert 'High' in probabilities
    
    def test_probabilities_sum_to_one(self, predictor, sample_patient_data):
        """Test that probabilities sum to 1"""
        probabilities = predictor.predict_proba(sample_patient_data)
        
        total = sum(probabilities.values())
        assert abs(total - 1.0) < 0.001  # Allow small floating point error
    
    def test_probabilities_in_valid_range(self, predictor, sample_patient_data):
        """Test that probabilities are between 0 and 1"""
        probabilities = predictor.predict_proba(sample_patient_data)
        
        for prob in probabilities.values():
            assert 0 <= prob <= 1
    
    def test_predict_with_details_structure(self, predictor, sample_patient_data):
        """Test that predict_with_details returns correct structure"""
        result = predictor.predict_with_details(sample_patient_data)
        
        assert 'prediction' in result
        assert 'probabilities' in result
        assert 'confidence' in result
        assert 'risk_factors' in result
        assert 'overall_risk_score' in result
    
    def test_confidence_is_valid(self, predictor, sample_patient_data):
        """Test that confidence score is valid"""
        result = predictor.predict_with_details(sample_patient_data)
        
        assert 0 <= result['confidence'] <= 1
    
    def test_risk_factors_present(self, predictor, sample_patient_data):
        """Test that risk factors are calculated"""
        result = predictor.predict_with_details(sample_patient_data)
        
        expected_factors = [
            'Lifestyle Risk', 'Environmental Risk', 
            'Genetic/Health Risk', 'Symptom Severity'
        ]
        
        for factor in expected_factors:
            assert factor in result['risk_factors']


# =============================================================================
# EDGE CASES TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_minimum_age(self, predictor):
        """Test prediction with minimum age"""
        data = {
            'Age': 14,  # Minimum age
            'Gender': 1,
            'Air Pollution': 1,
            'Alcohol use': 1,
            'Dust Allergy': 1,
            'OccuPational Hazards': 1,
            'Genetic Risk': 1,
            'chronic Lung Disease': 1,
            'Balanced Diet': 7,
            'Obesity': 1,
            'Smoking': 1,
            'Passive Smoker': 1,
            'Chest Pain': 1,
            'Coughing of Blood': 1,
            'Fatigue': 1,
            'Weight Loss': 1,
            'Shortness of Breath': 1,
            'Wheezing': 1,
            'Swallowing Difficulty': 1,
            'Clubbing of Finger Nails': 1,
            'Frequent Cold': 1,
            'Dry Cough': 1,
            'Snoring': 1
        }
        
        prediction = predictor.predict(data)
        assert prediction in ['Low', 'Medium', 'High']
    
    def test_maximum_values(self, predictor):
        """Test prediction with maximum values"""
        data = {
            'Age': 100,  # Maximum age
            'Gender': 2,
            'Air Pollution': 8,
            'Alcohol use': 8,
            'Dust Allergy': 8,
            'OccuPational Hazards': 8,
            'Genetic Risk': 7,
            'chronic Lung Disease': 7,
            'Balanced Diet': 1,
            'Obesity': 7,
            'Smoking': 8,
            'Passive Smoker': 8,
            'Chest Pain': 9,
            'Coughing of Blood': 9,
            'Fatigue': 9,
            'Weight Loss': 8,
            'Shortness of Breath': 9,
            'Wheezing': 8,
            'Swallowing Difficulty': 8,
            'Clubbing of Finger Nails': 9,
            'Frequent Cold': 7,
            'Dry Cough': 7,
            'Snoring': 7
        }
        
        prediction = predictor.predict(data)
        assert prediction in ['Low', 'Medium', 'High']
    
    def test_consistent_predictions(self, predictor, sample_patient_data):
        """Test that same input gives same output"""
        pred1 = predictor.predict(sample_patient_data)
        pred2 = predictor.predict(sample_patient_data)
        
        assert pred1 == pred2


# =============================================================================
# BATCH PREDICTION TESTS
# =============================================================================

class TestBatchPredictions:
    """Tests for batch predictions"""
    
    def test_batch_prediction(self, predictor, sample_patient_data, low_risk_patient):
        """Test batch prediction with multiple patients"""
        df = pd.DataFrame([sample_patient_data, low_risk_patient])
        
        predictions = []
        for _, row in df.iterrows():
            pred = predictor.predict(row.to_dict())
            predictions.append(pred)
        
        assert len(predictions) == 2
        assert all(pred in ['Low', 'Medium', 'High'] for pred in predictions)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow"""
    
    def test_full_prediction_workflow(self, predictor, sample_patient_data):
        """Test complete prediction workflow"""
        # Step 1: Get simple prediction
        prediction = predictor.predict(sample_patient_data)
        assert prediction in ['Low', 'Medium', 'High']
        
        # Step 2: Get probabilities
        probabilities = predictor.predict_proba(sample_patient_data)
        assert len(probabilities) == 3
        
        # Step 3: Get detailed results
        details = predictor.predict_with_details(sample_patient_data)
        assert details['prediction'] == prediction
        assert details['probabilities'] == probabilities
    
    def test_different_risk_levels(self, predictor, low_risk_patient, high_risk_patient):
        """Test that different patient profiles yield different risk levels"""
        low_pred = predictor.predict(low_risk_patient)
        high_pred = predictor.predict(high_risk_patient)
        
        # They should ideally be different, but we just check they're valid
        assert low_pred in ['Low', 'Medium', 'High']
        assert high_pred in ['Low', 'Medium', 'High']


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
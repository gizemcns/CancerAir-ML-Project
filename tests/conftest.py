"""
Pytest Configuration and Shared Fixtures
=========================================
Global test configuration and reusable fixtures(
Küresel test yapılandırması ve yeniden kullanılabilir armatürler)
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "inference: mark test for inference module"
    )
    config.addinivalue_line(
        "markers", "config: mark test for configuration"
    )


# =============================================================================
# SHARED FIXTURES - TEST DATA
# =============================================================================

@pytest.fixture(scope="session")
def sample_patient_data():
    """
    Sample patient data for testing (session scope - created once)
    
    Returns:
        dict: Patient data with all required features
    """
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


@pytest.fixture(scope="session")
def low_risk_patient():
    """
    Low risk patient data for testing
    
    Returns:
        dict: Patient data representing low risk
    """
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


@pytest.fixture(scope="session")
def medium_risk_patient():
    """
    Medium risk patient data for testing
    
    Returns:
        dict: Patient data representing medium risk
    """
    return {
        'Age': 45,
        'Gender': 2,
        'Air Pollution': 5,
        'Alcohol use': 4,
        'Dust Allergy': 4,
        'OccuPational Hazards': 4,
        'Genetic Risk': 4,
        'chronic Lung Disease': 3,
        'Balanced Diet': 4,
        'Obesity': 4,
        'Smoking': 4,
        'Passive Smoker': 3,
        'Chest Pain': 4,
        'Coughing of Blood': 3,
        'Fatigue': 4,
        'Weight Loss': 3,
        'Shortness of Breath': 4,
        'Wheezing': 3,
        'Swallowing Difficulty': 3,
        'Clubbing of Finger Nails': 2,
        'Frequent Cold': 3,
        'Dry Cough': 3,
        'Snoring': 3
    }


@pytest.fixture(scope="session")
def high_risk_patient():
    """
    High risk patient data for testing
    
    Returns:
        dict: Patient data representing high risk
    """
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


@pytest.fixture(scope="session")
def sample_dataframe(sample_patient_data):
    """
    Sample DataFrame for testing
    
    Args:
        sample_patient_data: Sample patient dictionary
        
    Returns:
        pd.DataFrame: DataFrame with sample patient
    """
    return pd.DataFrame([sample_patient_data])


@pytest.fixture(scope="session")
def multiple_patients_dataframe(low_risk_patient, medium_risk_patient, high_risk_patient):
    """
    DataFrame with multiple patients for batch testing
    
    Returns:
        pd.DataFrame: DataFrame with multiple patients
    """
    return pd.DataFrame([low_risk_patient, medium_risk_patient, high_risk_patient])


# =============================================================================
# SHARED FIXTURES - API PAYLOADS
# =============================================================================

@pytest.fixture(scope="session")
def api_patient_payload():
    """
    Patient data formatted for API requests
    
    Returns:
        dict: Patient data with underscore naming for API
    """
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


# =============================================================================
# SHARED FIXTURES - PATHS
# =============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """
    Test data directory path
    
    Returns:
        Path: Path to test data directory
    """
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def project_root():
    """
    Project root directory
    
    Returns:
        Path: Path to project root
    """
    return Path(__file__).parent.parent


# =============================================================================
# SHARED FIXTURES - VALIDATION
# =============================================================================

@pytest.fixture
def valid_feature_ranges():
    """
    Valid feature ranges for validation testing
    
    Returns:
        dict: Feature names and their valid ranges
    """
    return {
        'Age': (14, 100),
        'Gender': (1, 2),
        'Air Pollution': (1, 8),
        'Smoking': (1, 8),
        'Genetic Risk': (1, 7),
        'chronic Lung Disease': (1, 7),
        'Chest Pain': (1, 9),
        'Shortness of Breath': (1, 9)
    }


# =============================================================================
# HELPER FIXTURES
# =============================================================================

@pytest.fixture
def random_seed():
    """
    Random seed for reproducible tests
    
    Returns:
        int: Random seed value
    """
    return 42


@pytest.fixture
def setup_random_state(random_seed):
    """
    Setup random state for reproducible tests
    
    Args:
        random_seed: Random seed value
    """
    np.random.seed(random_seed)
    # Could also set other random states here
    yield
    # Cleanup if needed


# =============================================================================
# MOCK FIXTURES (for testing without models)
# =============================================================================

@pytest.fixture
def mock_prediction():
    """
    Mock prediction result
    
    Returns:
        str: Mock prediction
    """
    return "Medium"


@pytest.fixture
def mock_probabilities():
    """
    Mock probability distribution
    
    Returns:
        dict: Mock probabilities
    """
    return {
        'Low': 0.2,
        'Medium': 0.5,
        'High': 0.3
    }


@pytest.fixture
def mock_risk_factors():
    """
    Mock risk factors
    
    Returns:
        dict: Mock risk factors
    """
    return {
        'Lifestyle Risk': 5.5,
        'Environmental Risk': 6.0,
        'Genetic/Health Risk': 4.5,
        'Symptom Severity': 5.8,
        'Critical Symptoms': 2
    }


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture
def temp_model_dir(tmp_path):
    """
    Temporary directory for model files
    
    Args:
        tmp_path: Pytest's temporary path fixture
        
    Returns:
        Path: Temporary directory path
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


# =============================================================================
# PYTEST HOOKS
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically
    
    Args:
        config: Pytest config
        items: Collected test items
    """
    for item in items:
        # Add marker based on file name
        if "test_api" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        elif "test_inference" in str(item.fspath):
            item.add_marker(pytest.mark.inference)
        elif "test_config" in str(item.fspath):
            item.add_marker(pytest.mark.config)
        
        # Add marker based on test name
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


def pytest_report_header(config):
    """
    Add custom header to test report
    
    Args:
        config: Pytest config
        
    Returns:
        str: Header text
    """
    return """
    Cancer Risk Prediction - Test Suite
    ====================================
    Testing inference, API, and configuration modules
    """


# =============================================================================
# SESSION FIXTURES FOR PERFORMANCE
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def print_test_summary():
    """
    Print test summary at the end of session
    """
    yield
    print("\n" + "="*70)
    print("Test session completed!")
    print("="*70)


"""
Unit Tests for Configuration Module
====================================
Tests for config.py settings and functions(config.py ayarları ve işlevleri için testler)
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    pytest.skip("Config module not available", allow_module_level=True)


# =============================================================================
# PATH TESTS
# =============================================================================

class TestPaths:
    """Tests for path configurations"""
    
    def test_base_dir_exists(self):
        """Test that BASE_DIR is defined"""
        assert hasattr(config, 'BASE_DIR')
        assert isinstance(config.BASE_DIR, Path)
    
    def test_data_dir_defined(self):
        """Test that DATA_DIR is defined"""
        assert hasattr(config, 'DATA_DIR')
        assert isinstance(config.DATA_DIR, Path)
    
    def test_model_dir_defined(self):
        """Test that MODEL_DIR is defined"""
        assert hasattr(config, 'MODEL_DIR')
        assert isinstance(config.MODEL_DIR, Path)


# =============================================================================
# PARAMETER TESTS
# =============================================================================

class TestParameters:
    """Tests for model parameters"""
    
    def test_random_state_defined(self):
        """Test that RANDOM_STATE is defined"""
        assert hasattr(config, 'RANDOM_STATE')
        assert isinstance(config.RANDOM_STATE, int)
    
    def test_test_size_valid(self):
        """Test that TEST_SIZE is valid"""
        assert hasattr(config, 'TEST_SIZE')
        assert 0 < config.TEST_SIZE < 1
    
    def test_cv_folds_valid(self):
        """Test that CV_FOLDS is valid"""
        assert hasattr(config, 'CV_FOLDS')
        assert config.CV_FOLDS >= 2


# =============================================================================
# FEATURE TESTS
# =============================================================================

class TestFeatureConfiguration:
    """Tests for feature configurations"""
    
    def test_age_bins_defined(self):
        """Test that AGE_BINS is defined"""
        assert hasattr(config, 'AGE_BINS')
        assert isinstance(config.AGE_BINS, list)
        assert len(config.AGE_BINS) > 0
    
    def test_risk_weights_sum(self):
        """Test that risk weights sum to 1"""
        if hasattr(config, 'RISK_WEIGHTS'):
            total = sum(config.RISK_WEIGHTS.values())
            assert abs(total - 1.0) < 0.01


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidationFunctions:
    """Tests for validation functions"""
    
    def test_validate_feature_value_function_exists(self):
        """Test that validate_feature_value function exists"""
        assert hasattr(config, 'validate_feature_value')
        assert callable(config.validate_feature_value)
    
    def test_validate_feature_value_valid_input(self):
        """Test validation with valid input"""
        is_valid, message = config.validate_feature_value('Age', 50)
        assert is_valid is True
    
    def test_validate_feature_value_invalid_input(self):
        """Test validation with invalid input"""
        is_valid, message = config.validate_feature_value('Age', 150)
        assert is_valid is False


# =============================================================================
# BUSINESS RULES TESTS
# =============================================================================

class TestBusinessRules:
    """Tests for business rule configurations"""
    
    def test_critical_features_defined(self):
        """Test that critical features are defined"""
        assert hasattr(config, 'CRITICAL_FEATURES')
        assert isinstance(config.CRITICAL_FEATURES, list)
        assert len(config.CRITICAL_FEATURES) > 0
    
    def test_feature_ranges_defined(self):
        """Test that feature ranges are defined"""
        assert hasattr(config, 'FEATURE_RANGES')
        assert isinstance(config.FEATURE_RANGES, dict)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
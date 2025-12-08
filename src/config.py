
"""
Configuration file for Lung Cancer Prediction Project
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')

# Model paths
MODEL_PATH = os.path.join(MODELS_DIR, 'final_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, 'feature_names.pkl')

# Data file
DATA_FILE = 'cancer_patient_data_sets.csv'
DATA_PATH = os.path.join(RAW_DATA_DIR, DATA_FILE)

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

# Feature names (core features from dataset)
CORE_FEATURES = [
    'Age', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
    'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
    'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker',
    'Chest Pain', 'Coughing of Blood', 'Fatigue',
    'Weight Loss', 'Shortness of Breath', 'Wheezing',
    'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold',
    'Dry Cough', 'Snoring'
]

# Engineered features
ENGINEERED_FEATURES = [
    'smoke_alcohol_risk',
    'genetic_total_risk',
    'total_risk_score'
]

# Business thresholds
HIGH_RISK_THRESHOLD = 0.6
FALSE_NEGATIVE_TOLERANCE = 0.15

print("✅ Config loaded successfully!")
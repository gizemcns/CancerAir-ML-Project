"""
Configuration File - Cancer Risk Prediction
============================================
Konfigürasyon dosyaları, kodun kendisinden ayrı tutulan sabitleri ve değişken parametreleri içerir.
Merkezi konfigürasyon dosyası: paths, parameters, business rules
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS (Dosya Yolları)
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
NOTEBOOK_DIR = BASE_DIR / 'notebooks'
SRC_DIR = BASE_DIR / 'src'
DOCS_DIR = BASE_DIR / 'docs'

# C# Eğer yoksa dizinleri oluşturma
for directory in [DATA_DIR, MODEL_DIR, NOTEBOOK_DIR, SRC_DIR, DOCS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Veri yolları
RAW_DATA_PATH = DATA_DIR / 'cancer patient data sets.csv'
PROCESSED_DATA_PATH = DATA_DIR / 'cancer_data_feature_engineered.csv'

# Model yolları
FINAL_MODEL_PATH = MODEL_DIR / 'final_model.pkl'
FINAL_SCALER_PATH = MODEL_DIR / 'final_scaler.pkl'
FEATURE_LIST_PATH = MODEL_DIR / 'final_features.txt'

# =============================================================================
# MODEL PARAMETERS(MODEL PARAMETRELERİ)
# ==============================================================================
RANDOM_STATE = 42 # Tekrar üretilebilirlik için seed
TEST_SIZE = 0.2 # Test kümesi oranı
CV_FOLDS = 5  # 5 katlı çapraz doğrulama (5-fold cross-validation)

# Model hiperparametreleri (en iyi yapılandırma)
MODEL_PARAMS = {
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# =============================================================================
# FEATURE ENGINEERING PARAMETERS(ÖZELLİK MÜHENDİSLİĞİ PARAMETRELERİ)
# =============================================================================
# Age groups(Yaş grupları)
AGE_BINS = [0, 25, 40, 55, 100]
AGE_LABELS = ['Young', 'Adult', 'Middle_Aged', 'Senior']

# Smoking levels(Sigara içme seviyeleri)
SMOKING_BINS = [0, 2, 5, 10]
SMOKING_LABELS = ['Low', 'Medium', 'High']

# Pollution levels(Kirlilik seviyeleri)
POLLUTION_BINS = [0, 3, 6, 10]
POLLUTION_LABELS = ['Low', 'Medium', 'High']

# Critical symptom threshold(Kritik semptom eşiği)
CRITICAL_SYMPTOM_THRESHOLD = 6

# Risk score weights)( Risk puanı ağırlıkları)
RISK_WEIGHTS = {
    'environmental': 0.25,
    'lifestyle': 0.30,
    'genetic_health': 0.20,
    'symptom': 0.25
}

# =============================================================================
# BUSINESS RULES(İŞ KURALLARI)
# =============================================================================
# Risk level thresholds (0-1 scale)(Risk düzeyi eşikleri (0-1 ölçeği))
RISK_THRESHOLDS = {
    'low': 0.3,      # < 0.3 = Low risk
    'medium': 0.6,   # 0.3-0.6 = Medium risk
    'high': 0.6      # > 0.6 = High risk
}

# Critical features that must always be collected(Her zaman toplanması gereken kritik özellikler)
CRITICAL_FEATURES = [
    'Age',
    'Smoking',
    'Genetic Risk',
    'Air Pollution',
    'chronic Lung Disease',
    'Chest Pain',
    'Coughing of Blood',
    'Shortness of Breath'
]

# Feature value ranges (for validation)(Özellik değer aralıkları (doğrulama için))
FEATURE_RANGES = {
    'Age': (14, 100),
    'Gender': (1, 2),  # 1=Male, 2=Female
    'Air Pollution': (1, 8),
    'Alcohol use': (1, 8),
    'Dust Allergy': (1, 8),
    'OccuPational Hazards': (1, 8),
    'Genetic Risk': (1, 7),
    'chronic Lung Disease': (1, 7),
    'Balanced Diet': (1, 7),
    'Obesity': (1, 7),
    'Smoking': (1, 8),
    'Passive Smoker': (1, 8),
    'Chest Pain': (1, 9),
    'Coughing of Blood': (1, 9),
    'Fatigue': (1, 9),
    'Weight Loss': (1, 8),
    'Shortness of Breath': (1, 9),
    'Wheezing': (1, 8),
    'Swallowing Difficulty': (1, 8),
    'Clubbing of Finger Nails': (1, 9),
    'Frequent Cold': (1, 7),
    'Dry Cough': (1, 7),
    'Snoring': (1, 7)
}

# =============================================================================
# API CONFIGURATION(API YAPILANDIRMASI)
# =============================================================================
API_TITLE = "Cancer Risk Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Cancer Risk Level Prediction API

This API predicts cancer risk level (Low, Medium, High) based on patient data.

**Features:**
- Real-time predictions
- Batch predictions
- Health score calculation
- Risk factor analysis
"""

# API rate limiting(API oran sınırlaması)
API_RATE_LIMIT = "100/hour"

# =============================================================================
# MONITORING & LOGGING(İZLEME VE KAYIT)
# =============================================================================
# Metrics to track(İzlenecek metrikler)
MONITORING_METRICS = [
    'prediction_count',
    'average_response_time',
    'error_rate',
    'prediction_distribution',
    'feature_drift',
    'model_accuracy'
]

# Alert thresholds(Uyarı eşikleri)
ALERT_THRESHOLDS = {
    'error_rate': 0.05,  # Alert if > 5% errors
    'response_time': 2.0,  # Alert if > 2 seconds
    'prediction_drift': 0.1  # Alert if distribution shifts > 10%
}

# Logging configuration(Günlük kaydı yapılandırması)
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = BASE_DIR / 'logs' / 'app.log'

# =============================================================================
# MODEL RETRAINING TRIGGERS(MODEL YENİDEN EĞİTİM TETİKLEYİCİLERİ)
# =============================================================================
RETRAINING_TRIGGERS = {
    'performance_drop': 0.05,  # Retrain if accuracy drops > 5%
    'data_drift': 0.15,  # Retrain if data drift > 15%
    'time_based': 90,  # Retrain every 90 days
    'sample_count': 1000  # Retrain after 1000 new samples
}

# =============================================================================
# DEPLOYMENT CONFIGURATION( DAĞITIM YAPILANDIRMASI)
# =============================================================================
DEPLOYMENT_ENV = os.getenv('DEPLOYMENT_ENV', 'development')  # development, staging, production

DEPLOYMENT_CONFIGS = {
    'development': {
        'host': '0.0.0.0',
        'port': 8000,
        'debug': True,
        'reload': True
    },
    'staging': {
        'host': '0.0.0.0',
        'port': 8000,
        'debug': False,
        'reload': False
    },
    'production': {
        'host': '0.0.0.0',
        'port': 80,
        'debug': False,
        'reload': False
    }
}

# =============================================================================
# FEATURE SET (FINAL)(ÖZELLİK SETİ (SON))
# =============================================================================
FINAL_FEATURES = [
    'Smoking', 'Genetic Risk', 'Air Pollution', 'Alcohol use',
    'chronic Lung Disease', 'Age', 'Obesity', 'Chest Pain',
    'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath',
    'Wheezing', 'Passive Smoker', 'OccuPational Hazards',
    'Overall_Risk_Score', 'Lifestyle_Risk', 'Environmental_Risk',
    'Symptom_Severity', 'Respiratory_Score', 'Genetic_Health_Risk',
    'Smoking_Age_Interaction', 'Genetic_Age_Interaction',
    'Smoking_squared', 'Air Pollution_squared', 'Critical_Symptom_Count',
    'Age_Group', 'Smoking_Level'
]

# =============================================================================
# UTILITY FUNCTIONS(YARDIMCI FONKSİYONLAR)
# =============================================================================
def get_deployment_config():
    """Get deployment configuration based on environment"""
    return DEPLOYMENT_CONFIGS.get(DEPLOYMENT_ENV, DEPLOYMENT_CONFIGS['development'])

def validate_feature_value(feature_name, value):
    """Validate if feature value is in acceptable range"""
    if feature_name not in FEATURE_RANGES:
        return True, "Feature not in validation list"
    
    min_val, max_val = FEATURE_RANGES[feature_name]
    if min_val <= value <= max_val:
        return True, "Valid"
    else:
        return False, f"Value must be between {min_val} and {max_val}"

def get_risk_category(risk_score):
    """Convert risk score to risk category"""
    if risk_score < RISK_THRESHOLDS['low']:
        return 'Low'
    elif risk_score < RISK_THRESHOLDS['high']:
        return 'Medium'
    else:
        return 'High'

# =============================================================================
# VERSION INFO
# =============================================================================
__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

if __name__ == '__main__':
    print("="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"\nProject Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"\nDeployment Environment: {DEPLOYMENT_ENV}")
    print(f"Deployment Config: {get_deployment_config()}")
    print(f"\nFinal Feature Count: {len(FINAL_FEATURES)}")
    print(f"Critical Features: {len(CRITICAL_FEATURES)}")
    print("\n" + "="*80)
"""
FastAPI Application - Cancer Risk Prediction
=============================================
REST API for cancer risk prediction service
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import uvicorn
from datetime import datetime
import logging

# Import inference module
try:
    from inference import CancerRiskPredictor
except:
    print("⚠️ Warning: inference.py not found. Using mock predictor.")
    CancerRiskPredictor = None

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODELS
# =============================================================================
class PatientData(BaseModel):
    """Patient input data model"""
    Age: int = Field(..., ge=14, le=100, description="Patient age (14-100)")
    Gender: int = Field(..., ge=1, le=2, description="Gender (1=Male, 2=Female)")
    Air_Pollution: int = Field(..., ge=1, le=8, description="Air Pollution level (1-8)")
    Alcohol_use: int = Field(..., ge=1, le=8, description="Alcohol use level (1-8)")
    Dust_Allergy: int = Field(..., ge=1, le=8, description="Dust Allergy level (1-8)")
    OccuPational_Hazards: int = Field(..., ge=1, le=8, description="Occupational Hazards (1-8)")
    Genetic_Risk: int = Field(..., ge=1, le=7, description="Genetic Risk (1-7)")
    chronic_Lung_Disease: int = Field(..., ge=1, le=7, description="Chronic Lung Disease (1-7)")
    Balanced_Diet: int = Field(..., ge=1, le=7, description="Balanced Diet (1-7)")
    Obesity: int = Field(..., ge=1, le=7, description="Obesity level (1-7)")
    Smoking: int = Field(..., ge=1, le=8, description="Smoking level (1-8)")
    Passive_Smoker: int = Field(..., ge=1, le=8, description="Passive Smoker (1-8)")
    Chest_Pain: int = Field(..., ge=1, le=9, description="Chest Pain (1-9)")
    Coughing_of_Blood: int = Field(..., ge=1, le=9, description="Coughing of Blood (1-9)")
    Fatigue: int = Field(..., ge=1, le=9, description="Fatigue (1-9)")
    Weight_Loss: int = Field(..., ge=1, le=8, description="Weight Loss (1-8)")
    Shortness_of_Breath: int = Field(..., ge=1, le=9, description="Shortness of Breath (1-9)")
    Wheezing: int = Field(..., ge=1, le=8, description="Wheezing (1-8)")
    Swallowing_Difficulty: int = Field(..., ge=1, le=8, description="Swallowing Difficulty (1-8)")
    Clubbing_of_Finger_Nails: int = Field(..., ge=1, le=9, description="Clubbing of Finger Nails (1-9)")
    Frequent_Cold: int = Field(..., ge=1, le=7, description="Frequent Cold (1-7)")
    Dry_Cough: int = Field(..., ge=1, le=7, description="Dry Cough (1-7)")
    Snoring: int = Field(..., ge=1, le=7, description="Snoring (1-7)")
    
    class Config:
        schema_extra = {
            "example": {
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
        }
    
    def to_dict(self):
        """Convert to dictionary with proper column names"""
        return {
            'Age': self.Age,
            'Gender': self.Gender,
            'Air Pollution': self.Air_Pollution,
            'Alcohol use': self.Alcohol_use,
            'Dust Allergy': self.Dust_Allergy,
            'OccuPational Hazards': self.OccuPational_Hazards,
            'Genetic Risk': self.Genetic_Risk,
            'chronic Lung Disease': self.chronic_Lung_Disease,
            'Balanced Diet': self.Balanced_Diet,
            'Obesity': self.Obesity,
            'Smoking': self.Smoking,
            'Passive Smoker': self.Passive_Smoker,
            'Chest Pain': self.Chest_Pain,
            'Coughing of Blood': self.Coughing_of_Blood,
            'Fatigue': self.Fatigue,
            'Weight Loss': self.Weight_Loss,
            'Shortness of Breath': self.Shortness_of_Breath,
            'Wheezing': self.Wheezing,
            'Swallowing Difficulty': self.Swallowing_Difficulty,
            'Clubbing of Finger Nails': self.Clubbing_of_Finger_Nails,
            'Frequent Cold': self.Frequent_Cold,
            'Dry Cough': self.Dry_Cough,
            'Snoring': self.Snoring
        }

class PredictionResponse(BaseModel):
    """Prediction response model"""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    risk_factors: Dict[str, float]
    overall_risk_score: float
    timestamp: str
    recommendations: List[str]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    patients: List[PatientData]

# =============================================================================
# INITIALIZE APP
# =============================================================================
app = FastAPI(
    title="Cancer Risk Prediction API",
    description="Predict cancer risk level based on patient data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup"""
    global predictor
    try:
        if CancerRiskPredictor:
            predictor = CancerRiskPredictor()
            logger.info("✅ Predictor initialized successfully")
        else:
            logger.warning("⚠️ CancerRiskPredictor not available")
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {e}")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor is not None
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if predictor else "model not loaded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """
    Predict cancer risk level for a patient
    
    Args:
        patient: Patient data
        
    Returns:
        Prediction with probabilities and risk factors
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dictionary
        patient_dict = patient.to_dict()
        
        # Get prediction with details
        result = predictor.predict_with_details(patient_dict)
        
        # Generate recommendations based on risk factors
        recommendations = generate_recommendations(result['risk_factors'], result['prediction'])
        
        return {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities'],
            "risk_factors": result['risk_factors'],
            "overall_risk_score": result['overall_risk_score'],
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction for multiple patients
    
    Args:
        request: List of patient data
        
    Returns:
        List of predictions
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        
        for patient in request.patients:
            patient_dict = patient.to_dict()
            result = predictor.predict_with_details(patient_dict)
            recommendations = generate_recommendations(result['risk_factors'], result['prediction'])
            
            results.append({
                "prediction": result['prediction'],
                "confidence": result['confidence'],
                "probabilities": result['probabilities'],
                "risk_factors": result['risk_factors'],
                "overall_risk_score": result['overall_risk_score'],
                "recommendations": recommendations
            })
        
        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Random Forest Classifier",
        "features_count": 28,
        "classes": ["Low", "Medium", "High"],
        "version": "1.0.0",
        "trained_date": "2024",
        "framework": "scikit-learn"
    }

@app.get("/risk/factors")
async def risk_factors_info():
    """Get information about risk factors"""
    return {
        "risk_factors": {
            "Lifestyle Risk": {
                "description": "Combined score of smoking, alcohol, obesity, and diet",
                "range": [0, 10],
                "components": ["Smoking", "Alcohol use", "Obesity", "Balanced Diet"]
            },
            "Environmental Risk": {
                "description": "Combined score of environmental exposures",
                "range": [0, 8],
                "components": ["Air Pollution", "Dust Allergy", "Occupational Hazards"]
            },
            "Genetic/Health Risk": {
                "description": "Genetic predisposition and chronic conditions",
                "range": [0, 7],
                "components": ["Genetic Risk", "Chronic Lung Disease"]
            },
            "Symptom Severity": {
                "description": "Average severity of all symptoms",
                "range": [0, 9],
                "components": ["Chest Pain", "Coughing of Blood", "Fatigue", "Weight Loss", "etc."]
            },
            "Critical Symptoms": {
                "description": "Count of severe symptoms (≥6 level)",
                "range": [0, 4],
                "components": ["Chest Pain", "Coughing of Blood", "Weight Loss", "Shortness of Breath"]
            }
        }
    }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_recommendations(risk_factors: Dict[str, float], prediction: str) -> List[str]:
    """
    Generate personalized recommendations based on risk factors
    
    Args:
        risk_factors: Dictionary of risk factor scores
        prediction: Predicted risk level
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # General recommendation based on risk level
    if prediction == "High":
        recommendations.append("🚨 Seek immediate medical consultation for comprehensive cancer screening")
    elif prediction == "Medium":
        recommendations.append("⚠️ Schedule a medical check-up within the next month")
    else:
        recommendations.append("✅ Maintain regular health check-ups and healthy lifestyle")
    
    # Lifestyle recommendations
    if risk_factors['Lifestyle Risk'] > 6:
        recommendations.append("🚭 Consider smoking cessation programs and reduce alcohol consumption")
        recommendations.append("🏃 Adopt regular exercise routine and balanced diet")
    
    # Environmental recommendations
    if risk_factors['Environmental Risk'] > 6:
        recommendations.append("🏭 Minimize exposure to pollutants and use protective equipment at work")
        recommendations.append("🏠 Consider air purifiers for indoor air quality")
    
    # Symptom-based recommendations
    if risk_factors['Symptom Severity'] > 6:
        recommendations.append("🩺 Document all symptoms and discuss with healthcare provider")
    
    if risk_factors['Critical Symptoms'] >= 2:
        recommendations.append("🚑 Critical symptoms detected - seek immediate medical attention")
    
    # General health recommendations
    recommendations.append("💊 Follow prescribed medications and treatment plans")
    recommendations.append("📊 Monitor symptoms regularly and keep health records")
    
    return recommendations

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
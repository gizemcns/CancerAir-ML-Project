"""
Inference Script - Cancer Risk Prediction
==========================================
Inference Script (Çıkarım Betiği), bir makine öğrenimi modelinin eğitim aşaması tamamlandıktan sonra, 
bu modeli yeni, görülmemiş veriler üzerinde  çalıştırarak tahminler yapmasını sağlayan özel bir yazılım dosyasıdır.
Eğitilmiş modelden tahmin alma ve preprocessing bilgileri içermektedir.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import config
try:
    from config import *
except:
    # Fallback if config not available
    FINAL_MODEL_PATH = 'models/final_model.pkl'
    FINAL_SCALER_PATH = 'models/final_scaler.pkl'
    CRITICAL_SYMPTOM_THRESHOLD = 6

class CancerRiskPredictor:
    """
    Cancer Risk Level Predictor
    
    Loads trained model and provides prediction interface
    """
    
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model pickle file
            scaler_path: Path to trained scaler pickle file
        """
        self.model_path = model_path or FINAL_MODEL_PATH
        self.scaler_path = scaler_path or FINAL_SCALER_PATH
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        self.load_model()
    
    def load_model(self):
        """Load trained model and scaler"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Model loaded from {self.model_path}")
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Scaler loaded from {self.scaler_path}")
            
            # Get feature names from model if available
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_
            
        except FileNotFoundError as e:
            print(f"❌ Error: Model files not found. Please train the model first.")
            raise e
    
    def engineer_features(self, df):
        """
        Apply feature engineering to input data
        
        Args:
            df: Input DataFrame with raw features
            
        Returns:
            DataFrame with engineered features
        """
        df_fe = df.copy()
        
        # Age Groups
        df_fe['Age_Group'] = pd.cut(
            df_fe['Age'], 
            bins=[0, 25, 40, 55, 100], 
            labels=[0, 1, 2, 3]).astype(int)

        # Risk Scores
        df_fe['Environmental_Risk'] = (
            df_fe['Air Pollution'] + 
            df_fe['Dust Allergy'] + 
            df_fe['OccuPational Hazards']
        ) / 3
        
        df_fe['Lifestyle_Risk'] = (
            df_fe['Smoking'] + 
            df_fe['Alcohol use'] + 
            df_fe['Obesity'] +
            (10 - df_fe['Balanced Diet'])
        ) / 4
        
        df_fe['Genetic_Health_Risk'] = (
            df_fe['Genetic Risk'] + 
            df_fe['chronic Lung Disease']
        ) / 2
        
        # Symptom scores
        symptom_cols = ['Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
                        'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty']
        df_fe['Symptom_Severity'] = df_fe[symptom_cols].mean(axis=1)
        
        df_fe['Respiratory_Score'] = (
            df_fe['Shortness of Breath'] + 
            df_fe['Wheezing'] + 
            df_fe['Dry Cough'] +
            df_fe['chronic Lung Disease']
        ) / 4
        
        # Critical symptoms
        df_fe['Critical_Symptom_Count'] = (
            (df_fe['Chest Pain'] >= CRITICAL_SYMPTOM_THRESHOLD).astype(int) +
            (df_fe['Coughing of Blood'] >= CRITICAL_SYMPTOM_THRESHOLD).astype(int) +
            (df_fe['Weight Loss'] >= CRITICAL_SYMPTOM_THRESHOLD).astype(int) +
            (df_fe['Shortness of Breath'] >= CRITICAL_SYMPTOM_THRESHOLD).astype(int)
        )
        
        # Overall risk
        df_fe['Overall_Risk_Score'] = (
            df_fe['Environmental_Risk'] * 0.25 +
            df_fe['Lifestyle_Risk'] * 0.30 +
            df_fe['Genetic_Health_Risk'] * 0.20 +
            df_fe['Symptom_Severity'] * 0.25
        )
        
        # Interactions
        df_fe['Smoking_Age_Interaction'] = df_fe['Smoking'] * df_fe['Age']
        df_fe['Genetic_Age_Interaction'] = df_fe['Genetic Risk'] * df_fe['Age']
        df_fe['Smoking_Pollution'] = df_fe['Smoking'] * df_fe['Air Pollution']
        df_fe['Obesity_ChronicLung'] = df_fe['Obesity'] * df_fe['chronic Lung Disease']
        df_fe['PassiveSmoker_Pollution'] = df_fe['Passive Smoker'] * df_fe['Air Pollution']
        
        # Polynomial features
        for feat in ['Smoking', 'Air Pollution', 'Genetic Risk']:
            df_fe[f'{feat}_squared'] = df_fe[feat] ** 2
        
        # Binning
        df_fe['Smoking_Level'] = pd.cut(df_fe['Smoking'], 
        bins=[0, 2, 5, 10], 
        labels=[0, 1, 2]).astype(int)
        
        df_fe['Pollution_Level'] = pd.cut(df_fe['Air Pollution'], 
        bins=[0, 3, 6, 10], 
        labels=[0, 1, 2]).astype(int)

        return df_fe
    
    def preprocess(self, data):
        """
        Preprocess input data
        
        Args:
            data: Dict or DataFrame with patient data
            
        Returns:
            Preprocessed numpy array ready for prediction
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Engineer features
        df_fe = self.engineer_features(df)
        
        # Select final features
        final_features = [
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
        
        X = df_fe[final_features]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, data):
        """
        Make prediction
        
        Args:
            data: Dict or DataFrame with patient data
            
        Returns:
            Predicted risk level (Low, Medium, High)
        """
        X = self.preprocess(data)
        prediction = self.model.predict(X)
        return prediction[0]
    
    def predict_proba(self, data):
        """
        Get prediction probabilities
        
        Args:
            data: Dict or DataFrame with patient data
            
        Returns:
            Dict with probabilities for each class
        """
        X = self.preprocess(data)
        proba = self.model.predict_proba(X)[0]
        
        classes = self.model.classes_
        return {cls: prob for cls, prob in zip(classes, proba)}
    
    def predict_with_details(self, data):
        """
        Make prediction with detailed information
        
        Args:
            data: Dict or DataFrame with patient data
            
        Returns:
            Dict with prediction, probabilities, and risk factors
        """
        prediction = self.predict(data)
        probabilities = self.predict_proba(data)
        
        # Identify top risk factors
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        df_fe = self.engineer_features(df)
        
        risk_factors = {
            'Lifestyle Risk': df_fe['Lifestyle_Risk'].values[0],
            'Environmental Risk': df_fe['Environmental_Risk'].values[0],
            'Genetic/Health Risk': df_fe['Genetic_Health_Risk'].values[0],
            'Symptom Severity': df_fe['Symptom_Severity'].values[0],
            'Critical Symptoms': df_fe['Critical_Symptom_Count'].values[0]
        }
        
        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'confidence': max(probabilities.values()),
            'risk_factors': risk_factors,
            'overall_risk_score': df_fe['Overall_Risk_Score'].values[0]
        }

# =============================================================================
# EXAMPLE USAGE
# =============================================================================
def main():
    """Example usage of predictor"""
    
    print("="*80)
    print("CANCER RISK PREDICTOR - INFERENCE EXAMPLE")
    print("="*80)
    
    # Initialize predictor
    predictor = CancerRiskPredictor()
    
    # Example patient data
    patient_data = {
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
    
    print("\n📋 Patient Data:")
    print("-"*80)
    for key, value in list(patient_data.items())[:5]:
        print(f"   {key:30s}: {value}")
    print("   ...")
    
    # Simple prediction
    print("\n🔮 Making Prediction...")
    prediction = predictor.predict(patient_data)
    print(f"✅ Predicted Risk Level: {prediction}")
    
    # Prediction with probabilities
    print("\n📊 Prediction Probabilities:")
    probabilities = predictor.predict_proba(patient_data)
    for level, prob in probabilities.items():
        print(f"   {level:10s}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Detailed prediction
    print("\n📈 Detailed Analysis:")
    print("-"*80)
    details = predictor.predict_with_details(patient_data)
    
    print(f"\n🎯 Prediction: {details['prediction']}")
    print(f"🎲 Confidence: {details['confidence']:.4f} ({details['confidence']*100:.2f}%)")
    
    print(f"\n⚠️ Risk Factors:")
    for factor, score in details['risk_factors'].items():
        print(f"   {factor:25s}: {score:.2f}")
    
    print(f"\n💯 Overall Risk Score: {details['overall_risk_score']:.2f}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETED! ✅")
    print("="*80)

if __name__ == '__main__':
    main()
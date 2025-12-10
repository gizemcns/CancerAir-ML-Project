"""
Inference module for Lung Cancer Risk Prediction - FIXED VERSION
"""
import joblib
import pandas as pd
import numpy as np
from config import FINAL_MODEL_PATH as MODEL_PATH, FINAL_SCALER_PATH as SCALER_PATH, FEATURE_LIST_PATH as FEATURE_NAMES_PATH

class LungCancerPredictor:
    def __init__(self):
        """Initialize predictor with saved model and scaler"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load trained model, scaler and feature names"""
        
        """Eğitilmiş modeli, ölçekleyiciyi ve özellik adlarını yükle"""

        try:
        # load model
            self.model = joblib.load(MODEL_PATH)
            
            # load scaler
            self.scaler = joblib.load(SCALER_PATH)

            # load feature names from .txt file
            with open(FEATURE_NAMES_PATH, "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]

            print("✅ Model, scaler ve feature list başarıyla yüklendi!")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def prepare_features(self, input_data):
        """
        Prepare input data with feature engineering
        
        Args:
            input_data (dict): Raw input features
        
        Returns:
            pd.DataFrame: Prepared features
        """
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Feature Engineering (same as training)
        if 'Smoking' in df.columns and 'Alcohol use' in df.columns:
            df['smoke_alcohol_risk'] = df['Smoking'] * df['Alcohol use']
        
        if 'Genetic Risk' in df.columns and 'chronic Lung Disease' in df.columns:
            df['genetic_total_risk'] = df['Genetic Risk'] * df['chronic Lung Disease']
        
        if all(col in df.columns for col in ['Smoking', 'Alcohol use', 'Air Pollution', 'Genetic Risk']):
            df['total_risk_score'] = (
                df['Smoking'] + df['Alcohol use'] + 
                df['Air Pollution'] + df['Genetic Risk']
            ) / 4
        
        # Encode categorical if exists (pd.get_dummies)
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        # Align with training features
        for col in self.feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        df_encoded = df_encoded[self.feature_names]
        
        return df_encoded
    
    def predict(self, input_data):
        """
        Make prediction
        
        Args:
            input_data (dict): Input features
        
        Returns:
            tuple: (prediction, probability)
        """
        # Prepare features
        X = self.prepare_features(input_data)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return prediction, probability
    
    def predict_with_details(self, input_data):
        """
        Make prediction with detailed output
        
        Args:
            input_data (dict): Input features
        
        Returns:
            dict: Detailed prediction results
        """
        prediction, probability = self.predict(input_data)
        
        # Get class labels
        classes = self.model.classes_
        
        # Determine risk level (handle different class names)
        if str(prediction).lower() in ['high', '1', 'high risk']:
            risk_level = 'High'
        else:
            risk_level = 'Low'
        
        # Create probability dict with normalized keys
        prob_dict = {}
        for i, cls in enumerate(classes):
            if str(cls).lower() in ['high', '1', 'high risk']:
                prob_dict['High'] = float(probability[i])
            else:
                prob_dict['Low'] = float(probability[i])
        
        # Create detailed result
        result = {
            'prediction': str(prediction),
            'risk_level': risk_level,
            'probability': prob_dict,
            'confidence': float(max(probability)),
            'input_data': input_data
        }
        
        return result


# Test function
if __name__ == "__main__":
    # Sample input
    sample_input = {
        'Age': 45,
        'Air Pollution': 7,
        'Alcohol use': 3,
        'Dust Allergy': 5,
        'OccuPational Hazards': 4,
        'Genetic Risk': 6,
        'chronic Lung Disease': 3,
        'Balanced Diet': 4,
        'Obesity': 5,
        'Smoking': 8,
        'Passive Smoker': 6,
        'Chest Pain': 5,
        'Coughing of Blood': 2,
        'Fatigue': 6,
        'Weight Loss': 4,
        'Shortness of Breath': 5,
        'Wheezing': 4,
        'Swallowing Difficulty': 3,
        'Clubbing of Finger Nails': 2,
        'Frequent Cold': 4,
        'Dry Cough': 5,
        'Snoring': 3
    }
    
    # Initialize predictor
    try:
        predictor = LungCancerPredictor()
        
        # Make prediction
        result = predictor.predict_with_details(sample_input)
        
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Prediction: {result['prediction']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities: {result['probability']}")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nMake sure to run notebooks/06_pipeline.ipynb first!")
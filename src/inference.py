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
        
            # Feature list
            with open(FEATURE_NAMES_PATH, "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]

            # ⭐⭐ EKLEYECEĞİN SATIR TAM BURAYA ⭐⭐
            print("🔥 SCALER FEATURE LIST:", self.scaler.feature_names_in_)

            print("✅ Model, scaler ve feature list başarıyla yüklendi!")

        except Exception as e:
                print(f"❌ Error loading model: {e}")
                raise

        #     # load feature names from .txt file
        #     with open(FEATURE_NAMES_PATH, "r") as f:
        #         self.feature_names = [line.strip() for line in f.readlines()]

    
        #     print("\n🔥 SCALER GÖRDÜĞÜ FEATURE LİSTESİ:")

        #     print("SCALER FEATURES:", self.scaler.feature_names_in_)

        #     print("🔥 Listedeki kolon sayısı:", len(self.scaler.feature_names_in_))
            
        #     print("✅ Model, scaler ve feature list başarıyla yüklendi!")

        # except Exception as e:
        #     print(f"❌ Error loading model: {e}")
        #     raise
    
    def prepare_features(self, input_data):

        # -----------------------------------
        # 1) SAFE NUMERIC CONVERSION
        # -----------------------------------
        cleaned = {}
        for key, value in input_data.items():
            try:
                cleaned[key] = float(value)
            except:
                cleaned[key] = value
        input_data = cleaned

        df = pd.DataFrame([input_data])

        # -----------------------------------
        # 2) BASIC FEATURE ENGINEERING
        # -----------------------------------
        df['Smoking_squared'] = df['Smoking'] ** 2
        df['Air Pollution_squared'] = df['Air Pollution'] ** 2

        df['Age_Group'] = pd.cut(
            df['Age'], [0, 25, 40, 55, 100],
            labels=['Young', 'Adult', 'Middle_Aged', 'Senior']
        )

        df['Smoking_Level'] = pd.cut(
            df['Smoking'], [0, 2, 5, 10],
            labels=['Low', 'Medium', 'High']
        )

        # -----------------------------------
        # 3) ENVIRONMENTAL RISK
        # -----------------------------------
        df['Environmental_Risk'] = (
            df['Air Pollution'] +
            df['Dust Allergy'] +
            df['OccuPational Hazards']
        )

        # -----------------------------------
        # 4) LIFESTYLE RISK
        # -----------------------------------
        df['Lifestyle_Risk'] = (
            df['Smoking'] +
            df['Alcohol use'] +
            df['Obesity'] +
            (10 - df['Balanced Diet'])
        ) / 4

        # -----------------------------------
        # 5) GENETIC / HEALTH RISK
        # -----------------------------------
        df['Genetic_Health_Risk'] = (
            df['Genetic Risk'] +
            df['chronic Lung Disease']
        ) / 2

        # -----------------------------------
        # 6) SYMPTOM SEVERITY
        # -----------------------------------
        symptom_cols = [
            'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
            'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty'
        ]
        df['Symptom_Severity'] = df[symptom_cols].mean(axis=1)

        # -----------------------------------
        # 7) RESPIRATORY SCORE
        # -----------------------------------
        df['Respiratory_Score'] = (
            df['Shortness of Breath'] +
            df['Wheezing'] +
            df['Dry Cough'] +
            df['chronic Lung Disease']
        ) / 4

        # -----------------------------------
        # 8) CRITICAL SYMPTOM COUNT
        # -----------------------------------
        critical_threshold = 6
        df['Critical_Symptom_Count'] = (
            (df['Chest Pain'] >= critical_threshold).astype(int) +
            (df['Coughing of Blood'] >= critical_threshold).astype(int) +
            (df['Weight Loss'] >= critical_threshold).astype(int) +
            (df['Shortness of Breath'] >= critical_threshold).astype(int)
        )

        # -----------------------------------
        # 9) OVERALL RISK SCORE
        # -----------------------------------
        df['Overall_Risk_Score'] = (
            df['Environmental_Risk'] * 0.25 +
            df['Lifestyle_Risk'] * 0.30 +
            df['Genetic_Health_Risk'] * 0.20 +
            df['Symptom_Severity'] * 0.25
        )

        # -----------------------------------
        # 10) INTERACTION FEATURES
        # -----------------------------------
        df['Smoking_Age_Interaction'] = df['Smoking'] * df['Age']
        df['Genetic_Age_Interaction'] = df['Genetic Risk'] * df['Age']

        # -----------------------------------
        # 11) ONE-HOT ENCODING
        # -----------------------------------
        df_encoded = pd.get_dummies(df, drop_first=True)

        # -----------------------------------
        # 12) ALIGN COLUMNS (VERY IMPORTANT)
        # -----------------------------------
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
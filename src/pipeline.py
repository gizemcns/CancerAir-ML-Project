"""
Complete ML Pipeline - Cancer Risk Prediction
==============================================
Bu dosya tüm ML akışını tek script'te çalıştırmak için hazılanmıştır.

Stages:
1. Data loading
2. Feature engineering
3. Model training
4. Model evaluation
5. Model persistence
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import config
try:
    from config import *
except:
    print("⚠️ config.py not found. Using default values.")
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

class MLPipeline:
    """
    Complete Machine Learning Pipeline
    
    Handles data loading, preprocessing, training, and evaluation
    """
    
    def __init__(self, data_path, random_state=42):
        """
        Initialize pipeline
        
        Args:
            data_path: Path to raw data CSV
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        
        # Data
        self.df = None
        self.df_fe = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Models & preprocessing
        self.scaler = None
        self.model = None
        
        # Results
        self.results = {}
        
        print("="*80)
        print("ML PIPELINE INITIALIZED")
        print("="*80)
        print(f"Data Path: {data_path}")
        print(f"Random State: {random_state}")
    
    def load_data(self):
        """Load raw data"""
        print("\n" + "="*80)
        print("STAGE 1: DATA LOADING")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\n✅ Data loaded: {self.df.shape}")
        print(f"   Rows: {self.df.shape[0]:,}")
        print(f"   Columns: {self.df.shape[1]}")
        
        # Check target distribution
        print(f"\n📊 Target Distribution:")
        print(self.df['Level'].value_counts())
        
        return self
    
    def engineer_features(self):
        """Apply feature engineering"""
        print("\n" + "="*80)
        print("STAGE 2: FEATURE ENGINEERING")
        print("="*80)
        
        self.df_fe = self.df.copy()
        
        # Age Groups
        self.df_fe['Age_Group'] = pd.cut(
            self.df_fe['Age'], 
            bins=[0, 25, 40, 55, 100], 
            labels=[0, 1, 2, 3]
        ).astype(int)
        print("✅ Age_Group created")
        
        # Risk Scores
        self.df_fe['Environmental_Risk'] = (
            self.df_fe['Air Pollution'] + 
            self.df_fe['Dust Allergy'] + 
            self.df_fe['OccuPational Hazards']
        ) / 3
        print("✅ Environmental_Risk created")
        
        self.df_fe['Lifestyle_Risk'] = (
            self.df_fe['Smoking'] + 
            self.df_fe['Alcohol use'] + 
            self.df_fe['Obesity'] +
            (10 - self.df_fe['Balanced Diet'])
        ) / 4
        print("✅ Lifestyle_Risk created")
        
        self.df_fe['Genetic_Health_Risk'] = (
            self.df_fe['Genetic Risk'] + 
            self.df_fe['chronic Lung Disease']
        ) / 2
        print("✅ Genetic_Health_Risk created")
        
        # Symptom scores
        symptom_cols = ['Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
                        'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty']
        self.df_fe['Symptom_Severity'] = self.df_fe[symptom_cols].mean(axis=1)
        print("✅ Symptom_Severity created")
        
        self.df_fe['Respiratory_Score'] = (
            self.df_fe['Shortness of Breath'] + 
            self.df_fe['Wheezing'] + 
            self.df_fe['Dry Cough'] +
            self.df_fe['chronic Lung Disease']
        ) / 4
        print("✅ Respiratory_Score created")
        
        # Critical symptoms
        critical_threshold = 6
        self.df_fe['Critical_Symptom_Count'] = (
            (self.df_fe['Chest Pain'] >= critical_threshold).astype(int) +
            (self.df_fe['Coughing of Blood'] >= critical_threshold).astype(int) +
            (self.df_fe['Weight Loss'] >= critical_threshold).astype(int) +
            (self.df_fe['Shortness of Breath'] >= critical_threshold).astype(int)
        )
        print("✅ Critical_Symptom_Count created")
        
        # Overall risk
        self.df_fe['Overall_Risk_Score'] = (
            self.df_fe['Environmental_Risk'] * 0.25 +
            self.df_fe['Lifestyle_Risk'] * 0.30 +
            self.df_fe['Genetic_Health_Risk'] * 0.20 +
            self.df_fe['Symptom_Severity'] * 0.25
        )
        print("✅ Overall_Risk_Score created")
        
        # Interactions
        self.df_fe['Smoking_Age_Interaction'] = self.df_fe['Smoking'] * self.df_fe['Age']
        self.df_fe['Genetic_Age_Interaction'] = self.df_fe['Genetic Risk'] * self.df_fe['Age']
        self.df_fe['Smoking_Pollution'] = self.df_fe['Smoking'] * self.df_fe['Air Pollution']
        self.df_fe['Obesity_ChronicLung'] = self.df_fe['Obesity'] * self.df_fe['chronic Lung Disease']
        self.df_fe['PassiveSmoker_Pollution'] = self.df_fe['Passive Smoker'] * self.df_fe['Air Pollution']
        print("✅ Interaction features created (5)")
        
        # Polynomial features
        for feat in ['Smoking', 'Air Pollution', 'Genetic Risk']:
            self.df_fe[f'{feat}_squared'] = self.df_fe[feat] ** 2
        print("✅ Polynomial features created (3)")
        
        # Binning
        self.df_fe['Smoking_Level'] = pd.cut(
            self.df_fe['Smoking'], 
            bins=[0, 2, 5, 10], 
            labels=[0, 1, 2]
        ).astype(int)
        
        self.df_fe['Pollution_Level'] = pd.cut(
            self.df_fe['Air Pollution'], 
            bins=[0, 3, 6, 10], 
            labels=[0, 1, 2]
        ).astype(int)
        print("✅ Binning features created (2)")
        
        # Count new features
        original_cols = set(self.df.columns)
        new_cols = set(self.df_fe.columns) - original_cols
        print(f"\n📊 Feature Engineering Summary:")
        print(f"   Original features: {len(original_cols) - 3}")  # -3 for index, Patient Id, Level
        print(f"   New features: {len(new_cols)}")
        print(f"   Total features: {len(self.df_fe.columns) - 3}")
        
        return self
    
    def prepare_data(self):
        """Prepare train-test split"""
        print("\n" + "="*80)
        print("STAGE 3: DATA PREPARATION")
        print("="*80)
        
        # Define final features
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
        
        print(f"\n📝 Final feature count: {len(final_features)}")
        
        # Prepare X and y
        X = self.df_fe[final_features]
        y = self.df_fe['Level']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=self.random_state, 
            stratify=y
        )
        
        print(f"\n✅ Train-test split:")
        print(f"   Train: {self.X_train.shape[0]} samples")
        print(f"   Test:  {self.X_test.shape[0]} samples")
        
        # Scaling
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"✅ StandardScaler fitted")
        
        return self
    
    def train_model(self):
        """Train the model"""
        print("\n" + "="*80)
        print("STAGE 4: MODEL TRAINING")
        print("="*80)
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print(f"\n🔧 Model: {self.model.__class__.__name__}")
        print(f"⚙️ Parameters:")
        print(f"   - n_estimators: {self.model.n_estimators}")
        print(f"   - max_depth: {self.model.max_depth}")
        print(f"   - min_samples_split: {self.model.min_samples_split}")
        
        print(f"\n🚀 Training model...")
        start_time = datetime.now()
        
        self.model.fit(self.X_train_scaled, self.y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        return self
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "="*80)
        print("STAGE 5: MODEL EVALUATION")
        print("="*80)
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Accuracy scores
        train_acc = accuracy_score(self.y_train, y_train_pred)
        test_acc = accuracy_score(self.y_test, y_test_pred)
        
        # Cross-validation
        print(f"\n📊 Performing {CV_FOLDS}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, 
            self.X_train_scaled, 
            self.y_train, 
            cv=CV_FOLDS, 
            scoring='accuracy'
        )
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Train Accuracy:      {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"   CV Score (mean):     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   Overfitting:         {(train_acc - test_acc):.4f}")
        
        # Classification report
        print(f"\n📊 CLASSIFICATION REPORT:")
        print("="*80)
        print(classification_report(self.y_test, y_test_pred))
        
        # Confusion matrix
        print(f"\n📊 CONFUSION MATRIX:")
        print("="*80)
        cm = confusion_matrix(self.y_test, y_test_pred, labels=['Low', 'Medium', 'High'])
        cm_df = pd.DataFrame(
            cm,
            index=['True: Low', 'True: Medium', 'True: High'],
            columns=['Pred: Low', 'Pred: Medium', 'Pred: High']
        )
        print(cm_df)
        
        # Per-class accuracy
        print(f"\n📊 PER-CLASS ACCURACY:")
        for i, label in enumerate(['Low', 'Medium', 'High']):
            class_acc = cm[i, i] / cm[i, :].sum()
            print(f"   {label:10s}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        # Store results
        self.results = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'cv_score_mean': float(cv_scores.mean()),
            'cv_score_std': float(cv_scores.std()),
            'overfitting': float(train_acc - test_acc),
            'classification_report': classification_report(self.y_test, y_test_pred, output_dict=True),
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return self
    
    def save_artifacts(self, output_dir='models'):
        """Save model and artifacts"""
        print("\n" + "="*80)
        print("STAGE 6: SAVING ARTIFACTS")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save model
        model_path = output_path / 'final_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = output_path / 'final_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved: {scaler_path}")
        
        # Save feature names
        features_path = output_path / 'final_features.txt'
        with open(features_path, 'w') as f:
            for feat in self.X_train.columns:
                f.write(f"{feat}\n")
        print(f"✅ Features saved: {features_path}")
        
        # Save results
        results_path = output_path / 'model_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Results saved: {results_path}")
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = output_path / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        print(f"✅ Feature importance saved: {importance_path}")
        
        return self
    
    def run(self):
        """Run complete pipeline"""
        print("\n" + "🚀"*40)
        print("STARTING COMPLETE ML PIPELINE")
        print("🚀"*40)
        
        start_time = datetime.now()
        
        # Execute all stages
        self.load_data()
        self.engineer_features()
        self.prepare_data()
        self.train_model()
        self.evaluate_model()
        self.save_artifacts()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Final summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY! ✅")
        print("="*80)
        
        print(f"""
📊 FINAL RESULTS:
Test Accuracy:       {self.results['test_accuracy']:.4f} ({self.results['test_accuracy']*100:.2f}%)
CV Score:            {self.results['cv_score_mean']:.4f} ± {self.results['cv_score_std']:.4f}
Total Time:          {total_time:.2f} seconds

📦 SAVED ARTIFACTS:
✅ final_model.pkl
✅ final_scaler.pkl
✅ final_features.txt
✅ model_results.json
✅ feature_importance.csv

🚀 NEXT STEPS:
1. Review model results in models/model_results.json
2. Test inference with: python inference.py
3. Deploy API with: python app.py
4. Monitor model performance
    """)
        
        print("="*80)
        
        return self

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function"""
    
    # Configuration
    DATA_PATH = 'cancer patient data sets.csv'
    RANDOM_STATE = 42
    
    # Check if data exists
    if not Path(DATA_PATH).exists():
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        print(f"   Please place the data file in the current directory.")
        return
    
    # Initialize and run pipeline
    pipeline = MLPipeline(
        data_path=DATA_PATH,
        random_state=RANDOM_STATE
    )
    
    pipeline.run()

if __name__ == '__main__':
    main()
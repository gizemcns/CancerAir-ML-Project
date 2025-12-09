"""
Streamlit Web Application - Cancer Risk Prediction
===================================================
Interactive web interface for cancer risk prediction
Streamlit Web Uygulaması - Kanser Riski Tahmini, eğitilmiş makine öğrenimi modelini 
(Inference Script'te) alıp, kullanıcıların (doktorlar, araştırmacılar veya simülasyon yapanlar) bu modele 
kolayca veri girebileceği ve tahmin sonuçlarını anında
görsel olarak görebileceği interaktif bir web arayüzüdür. (user interface)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import predictor
try:
    from inference import CancerRiskPredictor
    PREDICTOR_AVAILABLE = True
except:
    PREDICTOR_AVAILABLE = False
    st.error("⚠️ Predictor not available. Please run pipeline.py first.")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Cancer Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================
if 'predictor' not in st.session_state and PREDICTOR_AVAILABLE:
    with st.spinner('Loading model...'):
        st.session_state.predictor = CancerRiskPredictor()
        st.success('✅ Model loaded successfully!')

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<h1 class="main-header">🏥 Cancer Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# =============================================================================
# SIDEBAR - INPUT FORM
# =============================================================================
st.sidebar.header("📋 Patient Information")
st.sidebar.markdown("Please enter patient details below:")

with st.sidebar.form("patient_form"):
    st.subheader("Demographics")
    age = st.number_input("Age", min_value=14, max_value=100, value=50, step=1)
    gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    
    st.subheader("Environmental Factors")
    air_pollution = st.slider("Air Pollution Level", 1, 8, 4)
    dust_allergy = st.slider("Dust Allergy", 1, 8, 4)
    occupational_hazards = st.slider("Occupational Hazards", 1, 8, 4)
    
    st.subheader("Lifestyle Factors")
    smoking = st.slider("Smoking Level", 1, 8, 4)
    passive_smoker = st.slider("Passive Smoker", 1, 8, 4)
    alcohol_use = st.slider("Alcohol Use", 1, 8, 4)
    obesity = st.slider("Obesity Level", 1, 7, 4)
    balanced_diet = st.slider("Balanced Diet", 1, 7, 4)
    
    st.subheader("Medical History")
    genetic_risk = st.slider("Genetic Risk", 1, 7, 4)
    chronic_lung_disease = st.slider("Chronic Lung Disease", 1, 7, 4)
    
    st.subheader("Symptoms")
    chest_pain = st.slider("Chest Pain", 1, 9, 4)
    coughing_of_blood = st.slider("Coughing of Blood", 1, 9, 4)
    fatigue = st.slider("Fatigue", 1, 9, 4)
    weight_loss = st.slider("Weight Loss", 1, 8, 4)
    shortness_of_breath = st.slider("Shortness of Breath", 1, 9, 4)
    wheezing = st.slider("Wheezing", 1, 8, 4)
    swallowing_difficulty = st.slider("Swallowing Difficulty", 1, 8, 4)
    clubbing_of_finger_nails = st.slider("Clubbing of Finger Nails", 1, 9, 4)
    frequent_cold = st.slider("Frequent Cold", 1, 7, 4)
    dry_cough = st.slider("Dry Cough", 1, 7, 4)
    snoring = st.slider("Snoring", 1, 7, 4)
    
    submit_button = st.form_submit_button("🔮 Predict Risk Level", use_container_width=True)

# =============================================================================
# MAIN CONTENT - PREDICTION
# =============================================================================
if submit_button and PREDICTOR_AVAILABLE:
    # Prepare patient data
    patient_data = {
        'Age': age,
        'Gender': gender,
        'Air Pollution': air_pollution,
        'Alcohol use': alcohol_use,
        'Dust Allergy': dust_allergy,
        'OccuPational Hazards': occupational_hazards,
        'Genetic Risk': genetic_risk,
        'chronic Lung Disease': chronic_lung_disease,
        'Balanced Diet': balanced_diet,
        'Obesity': obesity,
        'Smoking': smoking,
        'Passive Smoker': passive_smoker,
        'Chest Pain': chest_pain,
        'Coughing of Blood': coughing_of_blood,
        'Fatigue': fatigue,
        'Weight Loss': weight_loss,
        'Shortness of Breath': shortness_of_breath,
        'Wheezing': wheezing,
        'Swallowing Difficulty': swallowing_difficulty,
        'Clubbing of Finger Nails': clubbing_of_finger_nails,
        'Frequent Cold': frequent_cold,
        'Dry Cough': dry_cough,
        'Snoring': snoring
    }
    
    # Get prediction
    with st.spinner('Analyzing patient data...'):
        result = st.session_state.predictor.predict_with_details(patient_data)
    
    # Store in history
    st.session_state.prediction_history.append({
        'timestamp': datetime.now(),
        'prediction': result['prediction'],
        'confidence': result['confidence']
    })
    
    # Display results
    st.markdown("## 🎯 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Risk Level", result['prediction'])
    
    with col2:
        st.metric("Confidence", f"{result['confidence']*100:.1f}%")
    
    with col3:
        st.metric("Overall Risk Score", f"{result['overall_risk_score']:.2f}")
    
    # Risk level box
    risk_class = f"risk-{result['prediction'].lower()}"
    st.markdown(
        f'<div class="risk-box {risk_class}">Risk Level: {result["prediction"]}</div>',
        unsafe_allow_html=True
    )
    
    # Probability distribution
    st.markdown("### 📊 Probability Distribution")
    
    prob_df = pd.DataFrame({
        'Risk Level': list(result['probabilities'].keys()),
        'Probability': [v * 100 for v in result['probabilities'].values()]
    })
    
    fig_prob = px.bar(
        prob_df,
        x='Risk Level',
        y='Probability',
        color='Risk Level',
        color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
        text='Probability'
    )
    fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_prob.update_layout(showlegend=False, yaxis_title="Probability (%)")
    st.plotly_chart(fig_prob, use_container_width=True)
    
    # Risk factors analysis
    st.markdown("### ⚠️ Risk Factors Analysis")
    
    risk_factors_df = pd.DataFrame({
        'Risk Factor': list(result['risk_factors'].keys()),
        'Score': list(result['risk_factors'].values())
    })
    
    fig_risk = px.bar(
        risk_factors_df,
        x='Score',
        y='Risk Factor',
        orientation='h',
        color='Score',
        color_continuous_scale='Reds'
    )
    fig_risk.update_layout(showlegend=False, xaxis_title="Risk Score")
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Risk factors table
    st.dataframe(
        risk_factors_df.style.background_gradient(cmap='Reds', subset=['Score']),
        use_container_width=True
    )
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    if result['prediction'] == 'High':
        st.error("🚨 **Immediate Action Required**")
        st.markdown("""
        - Schedule immediate consultation with an oncologist
        - Undergo comprehensive cancer screening
        - Discuss family medical history with healthcare provider
        - Consider genetic counseling if genetic risk is high
        """)
    elif result['prediction'] == 'Medium':
        st.warning("⚠️ **Preventive Measures Recommended**")
        st.markdown("""
        - Schedule medical check-up within the next month
        - Adopt healthier lifestyle habits
        - Monitor symptoms regularly
        - Reduce exposure to environmental risk factors
        """)
    else:
        st.success("✅ **Maintain Healthy Lifestyle**")
        st.markdown("""
        - Continue regular health check-ups
        - Maintain balanced diet and exercise routine
        - Avoid smoking and limit alcohol consumption
        - Stay aware of any new symptoms
        """)
    
    # Detailed risk factor breakdown
    with st.expander("📈 Detailed Risk Factor Breakdown"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Lifestyle Factors**")
            st.write(f"- Smoking: {smoking}/8")
            st.write(f"- Alcohol: {alcohol_use}/8")
            st.write(f"- Obesity: {obesity}/7")
            st.write(f"- Diet: {balanced_diet}/7")
        
        with col2:
            st.markdown("**Environmental Factors**")
            st.write(f"- Air Pollution: {air_pollution}/8")
            st.write(f"- Dust Allergy: {dust_allergy}/8")
            st.write(f"- Occupational Hazards: {occupational_hazards}/8")
        
        st.markdown("**Medical Factors**")
        st.write(f"- Genetic Risk: {genetic_risk}/7")
        st.write(f"- Chronic Lung Disease: {chronic_lung_disease}/7")

# =============================================================================
# PREDICTION HISTORY
# =============================================================================
if st.session_state.prediction_history:
    st.markdown("---")
    st.markdown("## 📜 Prediction History")
    
    history_df = pd.DataFrame(st.session_state.prediction_history)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(
            history_df.style.format({
                'confidence': '{:.1%}',
                'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
            }),
            use_container_width=True
        )
    
    with col2:
        st.metric("Total Predictions", len(history_df))
        
        if len(history_df) > 0:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence*100:.1f}%")
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.experimental_rerun()

# =============================================================================
# SIDEBAR - INFO
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
This application predicts cancer risk levels based on:
- **Demographics**: Age, Gender
- **Environmental**: Pollution, Allergies, Hazards
- **Lifestyle**: Smoking, Alcohol, Diet, Obesity
- **Medical**: Genetic Risk, Chronic Conditions
- **Symptoms**: Various cancer-related symptoms

**Model**: Random Forest Classifier
**Accuracy**: ~XX% (test set)
""")

st.sidebar.markdown("### 📊 Model Info")
st.sidebar.markdown("""
- **Classes**: Low, Medium, High
- **Features**: 28 engineered features
- **Framework**: scikit-learn
- **Version**: 1.0.0
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>⚠️ <strong>Medical Disclaimer</strong>: This tool is for informational purposes only and should not replace professional medical advice.</p>
    <p>© 2024 Cancer Risk Prediction System | Powered by ML</p>
</div>
""", unsafe_allow_html=True)
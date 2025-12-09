"""
Streamlit Web Application for Lung Cancer Risk Prediction
Clean version - No FastAPI, No uvicorn
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from inference import LungCancerPredictor

# Page config
st.set_page_config(
    page_title="Lung Cancer Risk Predictor",
    page_icon="🫁",
    layout="wide"
)

# Initialize predictor
@st.cache_resource
def load_predictor():
    return LungCancerPredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Make sure to run notebooks/06_pipeline.ipynb first to generate model files!")
    st.stop()

# Title and description
st.title("🫁 Lung Cancer Risk Prediction System")
st.markdown("""
This application predicts lung cancer risk based on air pollution, lifestyle factors, and health indicators.
Enter patient information in the sidebar and click **Predict** to see the risk assessment.
""")

# Sidebar - Input Form
st.sidebar.header("📋 Patient Information")

# Demographic info
st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age", 18, 90, 45)

# Environmental factors
st.sidebar.subheader("Environmental Factors")
air_pollution = st.sidebar.slider("Air Pollution Level", 1, 10, 5, 
                                  help="1=Low, 10=High")
dust_allergy = st.sidebar.slider("Dust Allergy", 1, 10, 5)
occupational_hazards = st.sidebar.slider("Occupational Hazards", 1, 10, 4)

# Lifestyle factors
st.sidebar.subheader("Lifestyle Factors")
smoking = st.sidebar.slider("Smoking Level", 1, 10, 5, 
                            help="1=Non-smoker, 10=Heavy smoker")
passive_smoker = st.sidebar.slider("Passive Smoker Exposure", 1, 10, 4)
alcohol_use = st.sidebar.slider("Alcohol Use", 1, 10, 3)
balanced_diet = st.sidebar.slider("Balanced Diet", 1, 10, 5,
                                 help="1=Poor, 10=Excellent")
obesity = st.sidebar.slider("Obesity Level", 1, 10, 4)

# Health factors
st.sidebar.subheader("Health Indicators")
genetic_risk = st.sidebar.slider("Genetic Risk", 1, 10, 4)
chronic_lung_disease = st.sidebar.slider("Chronic Lung Disease", 1, 10, 3)

# Symptoms
st.sidebar.subheader("Symptoms")
chest_pain = st.sidebar.slider("Chest Pain", 1, 10, 4)
coughing_blood = st.sidebar.slider("Coughing of Blood", 1, 10, 2)
fatigue = st.sidebar.slider("Fatigue", 1, 10, 5)
weight_loss = st.sidebar.slider("Weight Loss", 1, 10, 3)
shortness_breath = st.sidebar.slider("Shortness of Breath", 1, 10, 4)
wheezing = st.sidebar.slider("Wheezing", 1, 10, 3)
swallowing_difficulty = st.sidebar.slider("Swallowing Difficulty", 1, 10, 2)
clubbing_nails = st.sidebar.slider("Clubbing of Finger Nails", 1, 10, 2)
frequent_cold = st.sidebar.slider("Frequent Cold", 1, 10, 4)
dry_cough = st.sidebar.slider("Dry Cough", 1, 10, 4)
snoring = st.sidebar.slider("Snoring", 1, 10, 3)

# Predict button
predict_button = st.sidebar.button("🔮 Predict Risk", type="primary")

# Main area
if predict_button:
    # Prepare input data
    input_data = {
        'Age': age,
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
        'Coughing of Blood': coughing_blood,
        'Fatigue': fatigue,
        'Weight Loss': weight_loss,
        'Shortness of Breath': shortness_breath,
        'Wheezing': wheezing,
        'Swallowing Difficulty': swallowing_difficulty,
        'Clubbing of Finger Nails': clubbing_nails,
        'Frequent Cold': frequent_cold,
        'Dry Cough': dry_cough,
        'Snoring': snoring
    }
    
    # Make prediction
    with st.spinner("Analyzing patient data..."):
        try:
            result = predictor.predict_with_details(input_data)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Please check that all required files are present and try again.")
            st.stop()
    
    # Display results
    st.header("📊 Prediction Results")
    
    # Risk level with color coding
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_level = result['risk_level']
        if risk_level == 'High':
            st.error(f"### ⚠️ {risk_level} Risk")
        else:
            st.success(f"### ✅ {risk_level} Risk")
    
    with col2:
        confidence = result['confidence']
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        prob_high = result['probability'].get('High', result['probability'].get('high', 0))
        st.metric("High Risk Probability", f"{prob_high:.1%}")
    
    # Probability visualization
    st.subheader("Risk Probability Distribution")
    
    prob_df = pd.DataFrame({
        'Risk Level': list(result['probability'].keys()),
        'Probability': list(result['probability'].values())
    })
    
    fig = go.Figure(data=[
        go.Bar(
            x=prob_df['Risk Level'],
            y=prob_df['Probability'],
            marker_color=['#00cc96' if p < 0.5 else '#ef553b' for p in prob_df['Probability']],
            text=[f"{p:.1%}" for p in prob_df['Probability']],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Probability by Risk Level",
        xaxis_title="Risk Level",
        yaxis_title="Probability",
        yaxis_tickformat='.0%',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors summary
    st.subheader("Key Risk Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**High Risk Factors:**")
        high_factors = []
        if smoking >= 7:
            high_factors.append(f"• Smoking: {smoking}/10")
        if air_pollution >= 7:
            high_factors.append(f"• Air Pollution: {air_pollution}/10")
        if genetic_risk >= 7:
            high_factors.append(f"• Genetic Risk: {genetic_risk}/10")
        if alcohol_use >= 7:
            high_factors.append(f"• Alcohol Use: {alcohol_use}/10")
        
        if high_factors:
            for factor in high_factors:
                st.markdown(factor)
        else:
            st.markdown("No major high-risk factors identified")
    
    with col2:
        st.markdown("**Protective Factors:**")
        protective = []
        if balanced_diet >= 7:
            protective.append(f"• Good Balanced Diet: {balanced_diet}/10")
        if smoking <= 3:
            protective.append(f"• Low Smoking: {smoking}/10")
        if obesity <= 3:
            protective.append(f"• Normal Weight: {obesity}/10")
        
        if protective:
            for factor in protective:
                st.markdown(factor)
        else:
            st.markdown("Limited protective factors")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    if risk_level == 'High':
        st.warning("""
        **⚠️ High Risk Detected:**
        - Consult with a healthcare professional immediately
        - Consider comprehensive screening tests
        - Reduce exposure to risk factors (smoking, pollution)
        - Improve lifestyle habits (diet, exercise)
        - Regular health monitoring recommended
        """)
    else:
        st.info("""
        **✅ Low Risk Detected:**
        - Maintain current healthy lifestyle
        - Continue regular health check-ups
        - Be mindful of environmental exposures
        - Early detection is key - monitor any symptoms
        """)

else:
    # Welcome message when no prediction made
    st.info("👈 Enter patient information in the sidebar and click **Predict Risk** to begin analysis.")
    
    # Show sample statistics
    st.subheader("📈 About This System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", "~85%")
    
    with col2:
        st.metric("Features Analyzed", "22+")
    
    with col3:
        st.metric("Risk Categories", "2")
    
    st.markdown("""
    ### How It Works
    
    This system uses machine learning (XGBoost) to analyze multiple factors:
    - **Environmental:** Air pollution, occupational hazards
    - **Lifestyle:** Smoking, alcohol use, diet, obesity
    - **Health:** Genetic risk, chronic conditions, symptoms
    
    The model was trained on patient data and validated to ensure reliable predictions.
    
    ### Important Note
    ⚠️ This tool is for informational purposes only and should not replace professional medical advice.
    Always consult with healthcare providers for proper diagnosis and treatment.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🫁 Lung Cancer Risk Prediction System | Zero2End ML Bootcamp 2024</p>
    <p>Built with Streamlit, XGBoost, and scikit-learn</p>
</div>
""", unsafe_allow_html=True)
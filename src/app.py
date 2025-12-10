"""
Streamlit Web Application for Lung Cancer Risk Prediction
Clean version - No FastAPI, No uvicorn
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from inference import LungCancerPredictor
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

st.markdown("""
<style>
.risk-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-top: 20px;
    transition: transform 0.3s ease-in-out;
}

.risk-card:hover {
    transform: scale(1.02);
}

.progress-container {
    width: 100%;
    background-color: #e0e0e0;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 15px;
}

.progress-bar {
    height: 22px;
    width: 0%;
    border-radius: 10px;
    animation: grow 1.5s ease-out forwards;
}
</style>
""", unsafe_allow_html=True)

def generate_pdf_report(result, input_data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # Başlık
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, 750, "Lung Cancer Risk Report")

    # Risk seviyesi
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 720, f"Risk Level: {result['risk_level']}")

    # Confidence
    c.setFont("Helvetica", 12)
    c.drawString(50, 700, f"Confidence: {result['confidence']*100:.2f}%")

    # Probability dağılımları
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 670, "Probability Distribution:")
    
    y = 650
    for k, v in result["probability"].items():
        c.setFont("Helvetica", 12)
        c.drawString(60, y, f"{k}: {v*100:.2f}%")
        y -= 20

    # Hasta girdileri
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y - 20, "Patient Inputs:")
    y -= 40

    for k, v in input_data.items():
        c.setFont("Helvetica", 10)
        c.drawString(60, y, f"{k}: {v}")
        y -= 15
        if y < 50:  # Sayfa dolarsa yenisini aç
            c.showPage()
            y = 750

    c.save()
    buffer.seek(0)
    return buffer


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
st.markdown("<div class='title'>🫁 Lung Cancer Risk Prediction System</div>", unsafe_allow_html=True)
st.markdown("""
This application predicts lung cancer risk based on air pollution, lifestyle factors, and health indicators.
Enter patient information in the sidebar and click **Predict** to see the risk assessment.
""")

# Sidebar - Input Form
st.sidebar.markdown("Use the sliders below to enter patient characteristics.")

# --- COLLAPSIBLE SIDEBAR SECTIONS ---

# Demographics
with st.sidebar.expander("🧍 Demographics", expanded=True):
    age = st.slider("Age", 18, 90, 45)
    gender = st.selectbox("Cinsiyet", options=[1, 2], format_func=lambda x: "Erkek" if x == 1 else "Kadın")

# Environmental Factors
with st.sidebar.expander("🌍 Environmental Factors", expanded=False):
    air_pollution = st.slider("Air Pollution Level", 1, 10, 5)
    dust_allergy = st.slider("Dust Allergy", 1, 10, 5)
    occupational_hazards = st.slider("Occupational Hazards", 1, 10, 4)

# Lifestyle
with st.sidebar.expander("🏃 Lifestyle Factors", expanded=False):
    smoking = st.slider("Smoking Level", 1, 10, 5)
    passive_smoker = st.slider("Passive Smoker Exposure", 1, 10, 4)
    alcohol_use = st.slider("Alcohol Use", 1, 10, 3)
    balanced_diet = st.slider("Balanced Diet", 1, 10, 5)
    obesity = st.slider("Obesity Level", 1, 10, 4)

# Health Indicators
with st.sidebar.expander("❤️ Health Indicators", expanded=False):
    genetic_risk = st.slider("Genetic Risk", 1, 10, 4)
    chronic_lung_disease = st.slider("Chronic Lung Disease", 1, 10, 3)

# Symptoms
with st.sidebar.expander("🤒 Symptoms", expanded=False):
    chest_pain = st.slider("Chest Pain", 1, 10, 4)
    coughing_blood = st.slider("Coughing of Blood", 1, 10, 2)
    fatigue = st.slider("Fatigue", 1, 10, 5)
    weight_loss = st.slider("Weight Loss", 1, 10, 3)
    shortness_breath = st.slider("Shortness of Breath", 1, 10, 4)
    wheezing = st.slider("Wheezing", 1, 10, 3)
    swallowing_difficulty = st.slider("Swallowing Difficulty", 1, 10, 2)
    clubbing_nails = st.slider("Clubbing of Finger Nails", 1, 10, 2)
    frequent_cold = st.slider("Frequent Cold", 1, 10, 4)
    dry_cough = st.slider("Dry Cough", 1, 10, 4)
    snoring = st.slider("Snoring", 1, 10, 3)

# Predict button remains outside expanders
predict_button = st.sidebar.button("🔮 Predict Risk", type="primary")


# Main area
if predict_button:
    # Prepare input data
    input_data = {
        'Age': age,
        'Gender': 1 if gender == "Male" else 2,
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

    # --- Animated Risk Card ---

    # Probabilities
    prob_high = result['probability'].get('High', result['probability'].get('high', 0))
    prob_low = result['probability'].get('Low', result['probability'].get('low', 0))

    # Bar rengi
    if result['risk_level'] == 'High':
        bar_color = "#e63946"   # Kırmızı
        percent = int(prob_high * 100)
    elif result['risk_level'] == 'Medium':
        bar_color = "#f1c40f"   # Sarı
        percent = 60            # Medium için stabil
    else:
        bar_color = "#2ecc71"   # Yeşil
        percent = int(prob_low * 100)

    # CSS içindeki placeholder'ı değiştir
    risk_css = f"""
    <style>
    @keyframes grow {{
        from {{ width: 0%; }}
        to {{ width: {percent}%; }}
    }}
    .progress-bar {{
        background-color: {bar_color};
    }}
    </style>
    """
    st.markdown(risk_css, unsafe_allow_html=True)

    # HTML Kart
    card_html = f"""
    <div class="risk-card">
        <h3 style="text-align:center;">Risk Level: {result['risk_level']}</h3>
        <div class="progress-container">
            <div class="progress-bar"></div>
        </div>
        <p style="text-align:center; margin-top:10px; font-size:16px;">
            Probability Score: {percent}%
        </p>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

        
    # PDF EXPORT BUTTON
    st.subheader("📄 Download Risk Report (PDF)")

    pdf_buffer = generate_pdf_report(result, input_data)

    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_buffer,
        file_name="lung_cancer_risk_report.pdf",
        mime="application/pdf"
    )


    # Risk level with color coding
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        risk_level = result['risk_level']
        if risk_level == 'High':
            st.error(f"### ⚠️ {risk_level} Risk")
        else:
            st.success(f"### ✅ {risk_level} Risk")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        confidence = result['confidence']
        st.metric("Confidence", f"{confidence:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        prob_high = result['probability'].get('High', 0)
        st.metric("High Risk Probability", f"{prob_high:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

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
    title="<b>Risk Probability Distribution</b>",
    title_x=0.3
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
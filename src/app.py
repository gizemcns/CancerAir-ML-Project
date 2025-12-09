import streamlit as st
import pandas as pd
import joblib
import os

# ------------------- MODEL YÜKLEME -------------------
@st.cache_resource
def load_model():
    model_path = "models/final_xgboost_model.joblib"
    if not os.path.exists(model_path):
        st.error("Model dosyası bulunamadı! Lütfen models/ klasörüne final_xgboost_model.joblib koyun.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# Sınıfların sırasını kontrol et (çok önemli!)
classes = model.classes_  # ['High', 'Low', 'Medium'] veya ['Low', 'Medium', 'High']

# ------------------- STREAMLIT UYGULAMA -------------------
st.set_page_config(page_title="Akciğer Kanseri Risk Tahmini", layout="centered")
st.title("Akciğer Kanseri Risk Tahmini")
st.markdown("### Hava kirliliği ve yaşam tarzı faktörlerine göre risk seviyenizi öğrenin")
st.info("Bu uygulama sadece bilgilendirme amaçlıdır. Tıbbi teşhis koyamaz.")

# ------------------- KULLANICI GİRİŞ FORMU -------------------
with st.sidebar:
    st.header("Hasta Bilgileri")
    
    Age = st.slider("Yaş", 1, 100, 50)
    Gender = st.selectbox("Cinsiyet", options=[1, 2], format_func=lambda x: "Erkek" if x == 1 else "Kadın")
    AirPollution = st.slider("Hava Kirliliği Seviyesi", 1, 10, 5)
    AlcoholUse = st.slider("Alkol Kullanımı", 1, 10, 5)
    DustAllergy = st.slider("Toz Alerjisi", 1, 10, 5)
    OccupationalHazards = st.slider("Mesleki Tehlikeler", 1, 10, 5)
    GeneticRisk = st.slider("Genetik Risk", 1, 10, 5)
    ChronicLungDisease = st.slider("Kronik Akciğer Hastalığı", 1, 10, 5)
    BalancedDiet = st.slider("Dengeli Beslenme", 1, 10, 5)
    Obesity = st.slider("Obezite", 1, 10, 5)
    Smoking = st.slider("Sigara Kullanımı", 1, 10, 5)
    PassiveSmoker = st.slider("Pasif İçicilik", 1, 10, 5)
    ChestPain = st.slider("Göğüs Ağrısı", 1, 10, 5)
    CoughingOfBlood = st.slider("Kanlı Öksürük", 1, 10, 5)
    Fatigue = st.slider("Yorgunluk", 1, 10, 5)
    WeightLoss = st.slider("Kilo Kaybı", 1, 10, 5)
    ShortnessOfBreath = st.slider("Nefes Darlığı", 1, 10, 5)
    Wheezing = st.slider("Hırıltılı Solunum", 1, 10, 5)
    SwallowingDifficulty = st.slider("Yutma Güçlüğü", 1, 10, 5)
    ClubbingOfFingerNails = st.slider("Parmak Çomaklaşması", 1, 10, 5)
    FrequentCold = st.slider("Sık Soğuk Algınlığı", 1, 10, 5)
    DryCough = st.slider("Kuru Öksürük", 1, 10, 5)
    Snoring = st.slider("Horlama", 1, 10, 5)

# ------------------- TAHMİN BUTONU -------------------
if st.button("RİSKİ HESAPLA", type="primary", use_container_width=True):
    with st.spinner("Tahmin yapılıyor..."):
        # 1. Kullanıcı verisini DataFrame'e çevir
        input_data = {
            'Age': Age, 'Gender': Gender, 'AirPollution': AirPollution,
            'AlcoholUse': AlcoholUse, 'DustAllergy': DustAllergy,
            'OccupationalHazards': OccupationalHazards, 'GeneticRisk': GeneticRisk,
            'ChronicLungDisease': ChronicLungDisease, 'BalancedDiet': BalancedDiet,
            'Obesity': Obesity, 'Smoking': Smoking, 'PassiveSmoker': PassiveSmoker,
            'ChestPain': ChestPain, 'CoughingOfBlood': CoughingOfBlood,
            'Fatigue': Fatigue, 'WeightLoss': WeightLoss, 'ShortnessOfBreath': ShortnessOfBreath,
            'Wheezing': Wheezing, 'SwallowingDifficulty': SwallowingDifficulty,
            'ClubbingOfFingerNails': ClubbingOfFingerNails, 'FrequentCold': FrequentCold,
            'DryCough': DryCough, 'Snoring': Snoring
        }
        
        df_input = pd.DataFrame([input_data])

        # 2. TAHMİN YAP
        prediction = model.predict(df_input)[0]                    # string veya int
        probabilities = model.predict_proba(df_input)[0]            # [0.1, 0.3, 0.6]

        # 3. Risk seviyesini string'e çevir
        risk_level = str(prediction)
        if risk_level.isdigit():
            risk_level = classes[int(prediction)]  # eğer model int döndürüyorsa

        # 4. Olasılıkları yüzdeye çevir
        prob_dict = {cls: round(prob * 100, 1) for cls, prob in zip(classes, probabilities)}
        max_prob = max(prob_dict.values())

        # ------------------- SONUÇ GÖSTER -------------------
        st.markdown("<br>", unsafe_allow_html=True)
        
        if risk_level == "High":
            st.error(f"### YÜKSEK RİSK DETEKLENDİ!")
            st.warning("Lütfen en kısa sürede bir doktora başvurun.")
        elif risk_level == "Medium":
            st.warning(f"### ORTA SEVİYE RİSK")
            st.info("Risk faktörlerinizi azaltmak için yaşam tarzı değişikliği önerilir.")
        else:
            st.success(f"### DÜŞÜK RİSK")
            st.balloons()

        # Olasılık çubuğu
        st.progress(max_prob / 100)
        st.write(f"**Tahmin: {risk_level}** → Low: {prob_dict.get('Low', 0)}% | Medium: {prob_dict.get('Medium', 0)}% | High: {prob_dict.get('High', 0)}%")

        # SHAP açıklama (isteğe bağlı)
        if st.checkbox("Detaylı Açıklama (SHAP) göster"):
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df_input)
                st_shap = st.empty()
                shap.initjs()
                st_shap.pyplot(shap.force_plot(explainer.expected_value[0], shap_values[0], df_input, matplotlib=True))
            except:
                st.info("SHAP görselleştirme için paket eksik.")

st.markdown("---")
st.caption("Geliştirici: Gizem Can Bayındır | Zero2End ML Bootcamp Final Projesi | 2025")
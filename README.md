"" 
# 🫁 Lung Cancer Risk Prediction System
## Hava Kirliliği ve Yaşam Tarzı Faktörleri ile Akciğer Kanseri Risk Tahmini

---

## 📌 Proje Hakkında

Bu proje, **Zero2End Machine Learning Bootcamp** kapsamında geliştirilmiş uçtan uca bir makine öğrenmesi uygulamasıdır. 

**Amaç:** Hava kirliliği, yaşam tarzı ve demografik faktörleri kullanarak bireylerdeki akciğer kanseri riskini tahmin eden bir ML modeli geliştirmek ve deploy etmek.

**Teslim Tarihi:** 9 Aralık 2025

---

## 🎯 Yapılacaklar Listesi

### ✅ Minimum Viable Product (MVP)

- [x] **Proje Kurulumu**
  - [x] GitHub repository oluşturma
  - [x] Klasör yapısı (data/, notebooks/, src/, models/)
  - [x] requirements.txt

- [ ] **Exploratory Data Analysis (EDA)**
  - [ ] Veri yükleme ve ilk inceleme
  - [ ] Missing value analizi
  - [ ] Değişken dağılımları ve korelasyonlar
  - [ ] EDA bulguları dokümantasyonu

- [ ] **Baseline Model**
  - [ ] Basit preprocessing pipeline
  - [ ] İlk model eğitimi (Logistic Regression/Decision Tree)
  - [ ] Baseline metrikler (Accuracy, F1-Score)

- [ ] **Feature Engineering**
  - [ ] Yeni feature türetme
  - [ ] Encoding ve scaling
  - [ ] Feature'ların model performansına etkisi

- [ ] **Model Optimization**
  - [ ] Çoklu model karşılaştırması (RF, XGBoost, LightGBM)
  - [ ] Hyperparameter tuning
  - [ ] Cross-validation

- [ ] **Model Evaluation**
  - [ ] Feature importance analizi
  - [ ] Confusion matrix & classification report
  - [ ] Final model seçimi

- [ ] **Pipeline Development**
  - [ ] End-to-end ML pipeline
  - [ ] Model serialization (pickle/joblib)
  - [ ] Inference scripti

- [ ] **Deployment**
  - [ ] Streamlit/Gradio arayüzü
  - [ ] REST API (FastAPI/Flask)
  - [ ] Cloud deployment (Streamlit Cloud/HuggingFace/Render)

- [ ] **Dokümantasyon**
  - [ ] README.md (bu dosya)
  - [ ] Notebook markdown hücreleri
  - [ ] 8 zorunlu soru cevapları

### 🌟 Bonus Özellikler (Olsa Güzel Olur)

- [ ] Düzenli Git commit geçmişi
- [ ] Monitoring dashboard
- [ ] Business kurgulu sistem tasarımı
- [ ] Üst yönetim sunumu (PPT/PDF)
- [ ] YouTube demo videosu
- [ ] Medium/Blog yazısı

---

## 🗂️ Proje Yapısı
```
lung-cancer-prediction/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_optimization.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_pipeline.ipynb
├── src/
│   ├── config.py
│   ├── inference.py
│   └── app.py
├── models/
│   └── final_model.pkl
├── docs/
└── tests/
```

---

## 📊 Veri Seti

**Kaynak:** [Kaggle - Cancer Patients and Air Pollution Dataset](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link)

**Özellikler:**
- Boyut: 10k+ satır, 10+ özellik
- Format: Tabular (.csv)
- Hedef Değişken: Akciğer kanseri varlığı (binary classification)
- Özellikler: Hava kirliliği seviyeleri, yaş, cinsiyet, sigara kullanımı, genetik risk faktörleri vb.

---

## 🛠️ Teknolojiler

- **Python 3.9+**
- **Veri Analizi:** Pandas, NumPy, Matplotlib, Seaborn
- **ML Kütüphaneleri:** Scikit-learn, XGBoost, LightGBM
- **Deployment:** Streamlit / FastAPI
- **Version Control:** Git, GitHub
- **Cloud:** Streamlit Cloud / HuggingFace Spaces

---

## 🚀 Kurulum

### Gereksinimler
```bash
git clone https://github.com/kullanici-adin/lung-cancer-prediction.git
cd lung-cancer-prediction
pip install -r requirements.txt
```

### Veri Hazırlama

1. Kaggle'dan veri setini indirin
2. `data/raw/` klasörüne yerleştirin

### Model Eğitimi
```bash
# Tüm pipeline'ı çalıştır
python src/pipeline.py
```

### Streamlit Uygulamasını Çalıştırma
```bash
streamlit run src/app.py
```

---

## 🌐 Demo

🔗 **Canlı Demo:** [Buraya deploy linki gelecek]

📸 **Ekran Görüntüsü:**
![Demo Screenshot](docs/screenshot.png)

---

## 📈 Sonuçlar (Güncellenecek)

| Metrik | Baseline | Final Model |
|--------|----------|-------------|
| Accuracy | - | - |
| Precision | - | - |
| Recall | - | - |
| F1-Score | - | - |

---

## 📝 Zorunlu Dokümantasyon Soruları

1. ✅ **Problem Tanımı:** [notebooks/01_eda.ipynb]
2. ✅ **Baseline Süreci:** [notebooks/02_baseline.ipynb]
3. ✅ **Feature Engineering:** [notebooks/03_feature_engineering.ipynb]
4. ✅ **Validasyon Şeması:** [notebooks/04_model_optimization.ipynb]
5. ✅ **Final Pipeline:** [notebooks/06_pipeline.ipynb]
6. ✅ **Model Karşılaştırma:** [notebooks/05_model_evaluation.ipynb]
7. ✅ **Business Uyumu:** [docs/business_alignment.md]
8. ✅ **Production Stratejisi:** [docs/production_strategy.md]

---

## 👤 İletişim

**Geliştirici:** [Gizem Can Bayındır]  
**Email:** [gizemcans2@gmail.com]  
**LinkedIn:** [linkedin.com/in/profil]  
**GitHub:** [@gizemcns](https://github.com/gizemcns)

---

## 📄 Lisans

Bu proje eğitim amaçlıdır ve Zero2End ML Bootcamp final projesi olarak geliştirilmiştir.

---

**Son Güncelleme:** 7 Aralık 2025

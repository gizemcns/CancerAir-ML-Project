
Cancer Risk Prediction - End-to-End ML Project
Kanser Riski Tahmini - Uçtan Uca ML Projesi

📌 **Proje Hakkında**

Bu proje, Zero2End Machine Learning Bootcamp kapsamında geliştirilmiş bir uçtan uca makine öğrenmesi uygulamasıdır. Hasta verilerine dayanarak, bireylerin akciğer kanseri risk seviyesini (Low, Medium, High) tahmin etmeyi amaçlamaktadır.

Amaç: Hava kirliliği, yaşam tarzı ve demografik faktörleri kullanarak bireylerdeki akciğer kanseri riskini tahmin eden bir ML modeli geliştirmek ve deploy etmek.

Teslim Tarihi: 9 Aralık 2025


✅ Minimum Viable Product (MVP)

**Proje Kurulumu**

  GitHub repository oluşturma

  Klasör yapısı (data/, notebooks/, src/, models/)

  requirements.txt

Exploratory Data Analysis (EDA)

  Veri yükleme ve ön inceleme

  Eksik değer analizi

  Değişken dağılımları ve korelasyonlar

  EDA bulguları dokümantasyonu

Baseline Model

  Basit preprocessing pipeline

  İlk model eğitimi (Logistic Regression / Decision Tree)

  Baseline metrikler (Accuracy, F1-Score)

Feature Engineering

  Yeni feature türetme

  Encoding ve scaling

  Feature’ların model performansına etkisi

Model Optimization

  Çoklu model karşılaştırması (Random Forest, XGBoost, LightGBM)

  Hyperparameter tuning

  Cross-validation

Model Evaluation

  Feature importance analizi

  Confusion matrix ve classification report

  Final model seçimi

Pipeline Development

  End-to-end ML pipeline

  Model serialization (pickle / joblib)

  Inference scripti

Deployment

  Streamlit / Gradio arayüzü

  REST API (FastAPI / Flask)

  Cloud deployment (Streamlit Cloud / HuggingFace / Render)

Dokümantasyon

  README.md (bu dosya)

  Notebook markdown hücreleri


🌟 **Bonus Özellikler**

 Düzenli Git commit geçmişi

 Monitoring dashboard

 Business kurgulu sistem tasarımı

🗂️ **Proje Yapısı**
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
│   ├── app.py
│   └── streamlit_app.py
├── models/
│   └── final_model.pkl
├── docs/
└── tests/

📊 **Veri Seti***

Kaynak: Kaggle - Cancer Patients and Air Pollution Dataset

**Özellikler:**

Boyut: 1000 satır × 24 sütun

Format: Tabular (.csv)

Hedef Değişken: Akciğer kanseri varlığı (level)

Özellikler: Hava kirliliği seviyeleri, yaş, cinsiyet, sigara kullanımı, genetik risk faktörleri vb.

Eksik Değer: Yok

Tekrar Eden Satır: 0

Veri Tipi: Tamamı sayısal (int64); kategorik değişkenler 1-9 arası skorlarla kodlanmış

🛠️ **Teknolojiler**

Python 3.9+

Veri Analizi: Pandas, NumPy, Matplotlib, Seaborn

ML Kütüphaneleri: Scikit-learn, XGBoost, LightGBM

Deployment: Streamlit / FastAPI

Version Control: Git, GitHub

Cloud: Streamlit Cloud / HuggingFace Spaces

🚀 **Kurulum**
git clone https://github.com/kullanici-adin/lung-cancer-prediction.git
cd lung-cancer-prediction
pip install -r requirements.txt

**Model Eğitimi:** [python src/pipeline.py](src/pipeline.py)


**Streamlit Uygulamasını Çalıştırma:** [streamlit run src/app.py](src/app.py)

🌐 **Demo**
[Canlı Demo:https://gizemcns-cancerair-ml-project-srcapp-featmvp-06c38v.streamlit.app/](https://gizemcns-cancerair-ml-project-srcapp-featmvp-06c38v.streamlit.app/)

📸 **Ekran Görüntüsü:** [Demo Screenshot docs/ekran-görüntüsü-2025-12-09%20 ](docs/ekran-görüntüsü-2025-12-09%20)


📝 Zorunlu Dokümantasyon Soruları

- **Problem Tanımı** → [01_EDA.ipynb](notebooks/01_eda.ipynb)  
- **Baseline Süreci** → [02_Baseline.ipynb](notebooks/02_baseline.ipynb)  
- **Feature Engineering** → [03_FeatureEngineering.ipynb](notebooks/03_feature_engineering.ipynb)  
- **Validasyon Şeması** → [04_ModelOptimization.ipynb](notebooks/04_model_optimization.ipynb)  
- **Final Pipeline** → [06_FinalPipeline.ipynb](notebooks/06_pipeline.ipynb)  
- **Model Karşılaştırma** → [05_Evaluation.ipynb](notebooks/05_model_evaluation.ipynb)  
- **Business Uyumu** → [docs/MD/business_alignment.md](docs/MD/business_alignment.md)  
- **Production Stratejisi** → [docs/production_strategy.md](docs/MD/production_strategy.md)  

👤 İletişim

Geliştirici: Gizem Can Bayındır

**Email:** [E-mail:gizemcans2@gmail.com](gizemcans2@gmail.com)

**LinkedIn:** [linkedin.com/in/gizemcanbayindir](https://www.linkedin.com/in/gizemcanbayindir)

**GitHub:** [Github:https://github.com/gizemcns](https://github.com/gizemcns)


📄 Lisans

Bu proje eğitim amaçlıdır ve Zero2End ML Bootcamp final projesi olarak geliştirilmiştir.

Son Güncelleme: 9 Aralık 2025


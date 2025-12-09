
Cancer Risk Prediction - End-to-End ML Project
Kanser Riski Tahmini - Uçtan Uca ML Projesi
📌 Proje Hakkında

Bu proje, Zero2End Machine Learning Bootcamp kapsamında geliştirilmiş bir uçtan uca makine öğrenmesi uygulamasıdır. Hasta verilerine dayanarak, bireylerin akciğer kanseri risk seviyesini (Low, Medium, High) tahmin etmeyi amaçlamaktadır.

Amaç: Hava kirliliği, yaşam tarzı ve demografik faktörleri kullanarak bireylerdeki akciğer kanseri riskini tahmin eden bir ML modeli geliştirmek ve deploy etmek.

Teslim Tarihi: 9 Aralık 2025

🎯 Yapılacaklar Listesi
✅ Minimum Viable Product (MVP)

Proje Kurulumu

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

 8 zorunlu soru-cevap

🌟 Bonus Özellikler

 Düzenli Git commit geçmişi

 Monitoring dashboard

 Business kurgulu sistem tasarımı

🗂️ Proje Yapısı
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

📊 Veri Seti

Kaynak: Kaggle - Cancer Patients and Air Pollution Dataset

Özellikler:

Boyut: 1000 satır × 24 sütun

Format: Tabular (.csv)

Hedef Değişken: Akciğer kanseri varlığı (level)

Özellikler: Hava kirliliği seviyeleri, yaş, cinsiyet, sigara kullanımı, genetik risk faktörleri vb.

Eksik Değer: Yok

Tekrar Eden Satır: 0

Veri Tipi: Tamamı sayısal (int64); kategorik değişkenler 1-9 arası skorlarla kodlanmış

🛠️ Teknolojiler

Python 3.9+

Veri Analizi: Pandas, NumPy, Matplotlib, Seaborn

ML Kütüphaneleri: Scikit-learn, XGBoost, LightGBM

Deployment: Streamlit / FastAPI

Version Control: Git, GitHub

Cloud: Streamlit Cloud / HuggingFace Spaces

🚀 Kurulum
git clone https://github.com/kullanici-adin/lung-cancer-prediction.git
cd lung-cancer-prediction
pip install -r requirements.txt

Model Eğitimi
python src/pipeline.py

Streamlit Uygulamasını Çalıştırma
streamlit run src/app.py

🌐 Demo

🔗 Canlı Demo: (https://gizemcns-cancerair-ml-project-srcapp-featmvp-06c38v.streamlit.app/)

📸 Ekran Görüntüsü: ![Demo Screenshot](docs/PNG/screenshot_2025_12_09_220639.png)


📝 Zorunlu Dokümantasyon Soruları

✅ Problem Tanımı: [notebooks/01_eda.ipynb]

✅ Baseline Süreci: [notebooks/02_baseline.ipynb]

✅ Feature Engineering: [notebooks/03_feature_engineering.ipynb]

✅ Validasyon Şeması: [notebooks/04_model_optimization.ipynb]

✅ Final Pipeline: [notebooks/06_pipeline.ipynb]

✅ Model Karşılaştırma: [notebooks/05_model_evaluation.ipynb]

✅ Business Uyumu: [docs/business_alignment.md]

✅ Production Stratejisi: [docs/production_strategy.md]

👤 İletişim

Geliştirici: Gizem Can Bayındır

Email: gizemcans2@gmail.com

LinkedIn: linkedin.com/in/gizemcanbayındırr

GitHub: @gizemcns

📄 Lisans

Bu proje eğitim amaçlıdır ve Zero2End ML Bootcamp final projesi olarak geliştirilmiştir.

Son Güncelleme: 9 Aralık 2025


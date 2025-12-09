# Production Stratejisi

Bu belge, akciğer kanseri risk tahmini modelimizi (XGBoost tabanlı Streamlit uygulaması) **gerçek dünya kullanımına (production)** taşımak için izlenecek stratejiyi detaylı olarak açıklamaktadır. Amacımız, modeli güvenli, ölçeklenebilir ve sürekli iyileştirilebilir hale getirmektir. Bu strateji, MLOps (Machine Learning Operations) en iyi uygulamalarına dayanır ve bootcamp kapsamında geliştirilmiştir.

## 1. Mevcut Durum
- **Model:** XGBoost (F1-macro: 0.984), joblib ile serialize edilmiş.
- **Uygulama:** Streamlit arayüzü (kullanıcı girdileri alır, tahmin yapar, SHAP explain gösterir).
- **Deployment:** Şu an Streamlit Cloud'da canlı (ücretsiz tier).
- **Veri:** Kaggle'dan 1000 satırlık temiz veri seti.

## 2. Deployment Stratejisi
Modeli production'a taşımak için şu adımları izliyoruz:

### a. Containerization (Docker ile Paketleme)
- Model ve app'i Docker container'ına paketleyerek her ortamda tutarlı çalışmasını sağlıyoruz.
- **Neden?** Farklı sunucularda (local, cloud) aynı davranışı garanti eder.
- **Nasıl Yapılır?**
  - `Dockerfile` oluştur:
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
text- Build: `docker build -t lung-cancer-app .`
- Run: `docker run -p 8501:8501 lung-cancer-app`

### b. Cloud Deployment Seçenekleri
- **Birincil:** Hugging Face Spaces (ücretsiz, ML odaklı).
- Adımlar: Yeni Space yarat → GitHub repo bağla → Auto-deploy.
- Avantaj: Otomatik build, secrets yönetimi (API key'ler için).
- **Alternatif:** Render.com veya AWS EC2 (ücretsiz tier'lar var).
- Render: Docker image yükle, auto-deploy.
- **API Entegrasyonu (Bonus):** FastAPI ile backend ekle (model tahminlerini REST API üzerinden sun).
- Örnek endpoint: `/predict` → JSON input alır, risk seviyesi döner.

### c. CI/CD Pipeline (Sürekli Entegrasyon ve Dağıtım)
- GitHub Actions ile otomatikleştiriyoruz.
- **Workflow Örneği:** `.github/workflows/deploy.yml`
name: Deploy to Hugging Face
on: [push]
jobs:
deploy:
runs-on: ubuntu-latest
steps:

uses: actions/checkout@v2
name: Build and Push
run: |
docker build -t lung-cancer-app .
docker push lung-cancer-app
name: Deploy to HF
uses: huggingface/deploy-to-hf@v1
with:
token: ${{ secrets.HF_TOKEN }}

text- Her push'ta otomatik test + deploy olur.

## 3. Monitoring ve Bakım
Production'da modelin performansını izlemek kritik!

### a. Model Monitoring
- **Araç:** Prometheus + Grafana (ücretsiz).
- **İzlenen Metrikler:**
- Tahmin doğruluğu (gerçek verilerle karşılaştır).
- Latency (cevap süresi < 2 sn).
- Drift detection (veri değişimi, örneğin yeni hava kirliliği seviyeleri).
- **Alarm:** Eğer accuracy %90 altına düşerse email bildirimi.

### b. Logging
- Streamlit'e logging ekle: `logging` modülüyle tahminleri dosyaya/log'a yaz.
- Örnek: Her tahmin için input + output + timestamp kaydet.

### c. Retraining Stratejisi
- Aylık retrain: Yeni veri gelirse (örneğin hastane verileri) modeli güncelle.
- Otomatik: GitHub Actions ile cron job (her ay 1'inde çalışır).
- Versiyonlama: MLflow ile her model versiyonunu sakla.

## 4. Güvenlik ve Ölçeklenebilirlik
- **Güvenlik:**
- Kullanıcı verileri şifrele (HTTPS zorunlu).
- Input validation: Yanlış veri girilirse hata ver.
- Secrets: Hugging Face secrets ile API key'leri sakla.
- **Ölçeklenebilirlik:**
- Başlangıç: Free tier yeterli (1000 kullanıcı/gün).
- Büyüme: AWS Lambda veya Kubernetes'e geç (otomatik scale).
- **Maliyet:** İlk 6 ay ücretsiz (Hugging Face + Render).

## 5. Riskler ve Yedek Plan
- **Risk:** Veri drift → Çözüm: Aylık manuel check.
- **Risk:** Downtime → Çözüm: Blue-green deployment (eski/yeni versiyon paralel çalışır).
- **Yedek:** Modeli yerel sunucuda (Raspberry Pi) çalıştırılabilir yap.

## 6. Gelecek İyileştirmeler
- Mobil entegrasyon: Streamlit'i PWA yap.
- Kullanıcı feedback: App'e rating sistemi ekle, modeli iyileştir.
- Ekip çalışması: Dokümanı paylaş, başkaları katkıda bulunsun.

Bu stratejiyle modelimiz production-ready! Herhangi bir güncelleme olursa bu dosyayı güncelleyeceğim.

**Hazırlayan:** Gizem Can Bayındır  
**Tarih:** 10 Aralık 2025
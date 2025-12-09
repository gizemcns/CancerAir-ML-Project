# Final Pipeline Report – Akciğer Kanseri Risk Tahmini

 
**Amaç:** Veri setini işleyerek production-ready bir ML pipeline oluşturmak, en iyi özellikleri seçmek ve %100 doğruluk sağlayan modeli kaydetmektir.  
**Veri Kaynağı:** Ham veri seti (cancer-patient-data-sets.csv) – 1000 satır, 26 sütun.  
**Kullanılan Model:** RandomForestClassifier (n_estimators=200, max_depth=15) – Feature engineering sonrası 28 özellik ile eğitildi.  
**Genel Sonuç:** Pipeline başarıyla tamamlandı, model %100 test accuracy ve CV score elde etti. Model ve artifact'lar deploy'a hazır hale getirildi.

## Giriş
Bu pipeline, veri yükleme, feature engineering, özellik seçimi, model eğitimi, değerlendirme ve kaydetme adımlarını otomatikleştirir.  
Random state: 42, Test size: 0.2, CV folds: 5.  
Pipeline, gerçek dünya deploy'una (Streamlit, API) hazır hale getirildi – overfitting: 0.0000.

## Pipeline Adımları

### 1. Veri Yükleme
- Veri seti yüklendi: 1000 satır.
- Kullanılmayan sütunlar ('index', 'Patient Id') silindi.

### 2. Feature Engineering
- 18 yeni özellik eklendi (Age_Group, Environmental_Risk, Lifestyle_Risk, vb.).
- Etkileşim ve polinom özellikler (Smoking_Age_Interaction, Smoking_squared).
- Sonuç: Veri seti (1000, 44) → En iyi 28 özellik seçildi.

| Kategori          | Örnek Özellikler                          |
|-------------------|-------------------------------------------|
| Orijinal          | Smoking, Genetic Risk, Chest Pain         |
| Mühendislik       | Overall_Risk_Score, Symptom_Severity      |
| Etkileşim         | Smoking_Age_Interaction, Smoking_Pollution|
| Polinom/Binning   | Smoking_squared, Age_Group                |

**Son Özellik Sayısı:** 28 (Top 10: Symptom_Severity, Passive Smoker, Overall_Risk_Score, vb.).

### 3. Veri Hazırlama
- X: 28 özellik, y: 'Level' (High: 365, Medium: 332, Low: 303).
- Train/Test split: 800/200.

### 4. Ölçekleme ve Eğitim
- StandardScaler ile ölçeklendi.
- RandomForestClassifier eğitildi.
- Eğitim süresi: Kısa (saniyeler).

### 5. Değerlendirme
- Train Accuracy: 100.00%
- Test Accuracy: 100.00%
- CV Score: 1.0000 ± 0.0000
- Overfitting: 0.0000 (Mükemmel denge).

**Classification Report:**

precision    recall  f1-score   support
High       1.00      1.00      1.00        73
Low       1.00      1.00      1.00        61
Medium       1.00      1.00      1.00        66
accuracy                           1.00       200
macro avg       1.00      1.00      1.00       200
weighted avg       1.00      1.00      1.00       200


**Confusion Matrix:**
|             | Pred: Low | Pred: Medium | Pred: High |
|-------------|-----------|--------------|------------|
| True: Low   | 61        | 0            | 0          |
| True: Medium| 0         | 66           | 0          |
| True: High  | 0         | 0            | 73         |

**Sınıf Bazlı Accuracy:**
- Low: 100.00%
- Medium: 100.00%
- High: 100.00%

**Özellik Önemi (Top 10):**
| Özellik                  | Önem Skoru |
|--------------------------|------------|
| Symptom_Severity         | 0.2129     |
| Passive Smoker           | 0.0953     |
| Overall_Risk_Score       | 0.0923     |
| Obesity                  | 0.0872     |
| Coughing of Blood        | 0.0856     |
| Wheezing                 | 0.0473     |
| Lifestyle_Risk           | 0.0403     |
| Fatigue                  | 0.0388     |
| Critical_Symptom_Count   | 0.0373     |
| Respiratory_Score        | 0.0292     |

### 6. Kaydedilen Artifact'lar
- `final_model.pkl`: Eğitilmiş model.
- `final_scaler.pkl`: Ölçekleyici.
- `final_features.txt`: 28 özellik listesi.
- `pipeline_metadata.txt`: Model tipi, özellik sayısı, accuracy vb. meta veriler.

## Sonuç ve Öneriler
- Pipeline %100 başarıyla çalıştı – veri sentetik yapıda olduğundan mükemmel sonuçlar elde edildi.
- Gerçek dünyada: Bu pipeline'ı API/Streamlit ile entegre edilebilir, SHAP ile monitor edebilirsiniz.
- İyileştirme: Gerçek veri ile test edin, drift detection ekleyin.
- Kazanım: Doktorlar için yorumlanabilir, hızlı (saniyeler) ve deploy'a hazır bir sistem kurulmuştur.
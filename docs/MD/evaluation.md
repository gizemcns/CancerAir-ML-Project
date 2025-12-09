# Model Evaluation Report – Akciğer Kanseri Risk Tahmini

**Amaç:** Bu aşamanın temel amacı, Model Optimizasyonu'nda seçilen en iyi modelin (Random Forest) tahminlerini **yorumlamak (interpretability)**, modelin kararlarına en çok etki eden **özellikleri (feature) belirlemek** ve nihai dağıtım (deployment) için en yüksek performansı koruyarak en sade (optimal) özellik alt kümesini oluşturmaktır. 
**Veri Kaynağı:** Feature engineering sonrası zenginleştirilmiş veri seti (41 özellik).  
**Kullanılan Model:** RandomForestClassifier (n_estimators=200, max_depth=15) – %100 doğruluk sağlayan model.

## Giriş
Bu raporda, modelin karar verme sürecini anlamak için:
- Özellik önemi (Feature Importance) hesaplandı.
- En iyi özellik sayısı belirlenerek model basitleştirildi (underfitting/overfitting dengesi korunarak).
- İş kritik özellikleri (business critical) kontrol edildi.

Sonuç: Model zaten %100 doğruluk verdiği için odak, **yorumlanabilirlik ve verimlilik** üzerindeydi.

## Yöntemler
### 1. Özellik Önemi (Tree-Based)
- Random Forest modeli ile özellik önemleri hesaplandı.
- Sıralama: Gini impurity azalmasına göre.

### 2. Özellik Seçimi.
- Optimal k: Çapraz doğrulama (CV) ile belirlenen en iyi özellik sayısı (örneğin, 10 özellik şeklinde kontrol edildi.

### 3. İş Kritik Özellik Kontrolü
- Tanımlı kritik özellikler: Coughing of Blood, Chest Pain, Genetic Risk, Smoking, Air Pollution.

## Sonuçlar

### Özellik Önemi Tablosu (Top 10)
| Sıra | Özellik              | Önem Skoru |
|------|----------------------|------------|
| 1    | Symptom_Severity     | 0.1651     |
| 2    | Overall_Risk_Score   | 0.08384    |
| 3    | Obesity_ChronicLung  | 0.07153    |
| 4    | Passive Smoker       | 0.063785   |
| 5    | Coughing of Blood    | 0.059823   |
| 6    | Obesity              | 0.052915   |
| 7    | Smoking_Pollution    | 0.040482   |
| 8    | Critical_Symptom_Count| 0.035828  |
| 9    | Wheezing              |  0.032950 |
| 10   | Lifestyle_Risk        | 0.027722  |


**Yorum:** Belirti şiddeti(Symptom_Severity) ve genel risk skoru(Overall_Risk_Score), kanser riskini en çok etkileyen faktörler.

### Özellik Seçimi Sonuçları
- Optimal Özellik Sayısı: **10** (CV score: 1.0000, Test accuracy: 1.0000).
- Seçilen Özellikler:
  1. Symptom_Severity
  2. Overall_Risk_Score
  3. Obesity_ChronicLung
  4. Passive Smoker
  5. Coughing of Blood
  6. Obesity
  7. Smoking_Pollution
  8. Critical_Symptom_Count
  9. Wheezing
  10. Lifestyle_Risk

**Performans Grafiği:** Farklı k değerleri için accuracy eğrisi `feature_selection_curve.png` dosyasında.

### İş Kritik Özellik Kontrolü

   ⚠️ Smoking - NOT IN FINAL SET
   ⚠️ Age - NOT IN FINAL SET
   ⚠️ Genetic Risk - NOT IN FINAL SET
   ⚠️ Air Pollution - NOT IN FINAL SET
   ⚠️ Chest Pain - NOT IN FINAL SET
   ✅ Coughing of Blood
   ⚠️ Shortness of Breath - NOT IN FINAL SET
   ⚠️ chronic Lung Disease - NOT IN FINAL SET

Coughing of Blood özelliği dışında kalanlar dahil edilmedi.

**Yorum:** Tüm kritik özellikler final sette yer almıyor – model klinik olarak güvenilir görünmüyor.

## Öneriler
- **Final Model:** 10 özellikli versiyonu kullanın (daha az bellek, daha hızlı tahmin).
- **Underfitting/Overfitting:** Veri sentetik olduğu için yok, ancak gerçek veride SHAP ile monitor kontrol edilebilir.
- **Klinik Uygulama:** Top 5 risk faktörünü doktorlar için dashboard'a ekleyebilirsiniz.
- **Sonraki Adım:** Bu özelliklerle Streamlit app geliştirin.

## Çıktılar
- `feature_importance.csv`: Özellik önem tablosu.
- `final_feature_set.txt`: Seçilen özellikler ve kritik özellik kontrolü.
- `feature_selection_results.csv`: Farklı k değerleri için performans.
- Grafikler: `feature_importance.png`.


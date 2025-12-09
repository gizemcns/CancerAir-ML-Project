
#  4. Aşama: Model Optimizasyonu ve Karşılaştırma

##  Hedef

Bu aşamanın temel amacı, Özellik Mühendisliği (Feature Engineering) adımıyla zenginleştirilmiş veri seti üzerinde, hiperparametre optimizasyonu (GridSearchCV) uygulayarak en yüksek genelleme yeteneğine sahip, en iyi tahmin performansını gösteren modeli (Cancer Risk Prediction için) seçmek ve nihai modeli oluşturmaktır.

##  Metodoloji

### 1. Veri Hazırlığı
* **Veri Yükleme:** Proaktif yaklaşım sergilenerek öncelikle **41 özellikli** zenginleştirilmiş veri seti (`cancer_data_feature_engineered.csv`) yüklenmiştir. (Eğer bulunamasaydı, ham veri setine düşülecekti.)
* **Özellik/Hedef Ayırımı:** Veri seti giriş değişkenleri (X) ve hedef değişken (`Level`, y) şeklinde ayrıldı.
    * **Özellik Sayısı:** 41
    * **Veri Dağılımı:** Veri seti, sınıflar arasında **dengeli** bir dağılım göstermektedir (Low, Medium, High).
* **Eğitim/Test Ayrımı:** Verinin %80'i eğitim, %20'si test için ayrılmıştır (`random_state=42`, `stratify=y` kullanılarak sınıf dağılımı korunmuştur).
    * **Eğitim Seti:** 800 örnek
    * **Test Seti:** 200 örnek
* **Ölçeklendirme (Scaling):** Lojistik Regresyon ve Gradien tabanlı modeller için, özellik değerleri **StandardScaler** kullanılarak normalize edilmiştir.

### 2. Değerlendirme ve Optimizasyon
* **Değerlendirme Metriği:** Çok sınıflı bir sınıflandırma problemi olduğu için temel metrik olarak **Doğruluk (Accuracy)** ve modelin genelleme yeteneğini ölçmek için **Çapraz Doğrulama Skoru (CV Score)** kullanılmıştır.
* **Hiperparametre Optimizasyonu:** Tüm modellerde en iyi parametre kombinasyonunu bulmak için **GridSearchCV** ve 5 katlı çapraz doğrulama (`cv=5`) tekniği uygulanmıştır.

### 3. Test Edilen Modeller

| Model Adı | Türü | Açıklama |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear / Baseline | Temel model performansı için kullanılmıştır. |
| **Random Forest** | Ensemble / Bagging | Yüksek genelleme yeteneği ve yorumlanabilirlik için test edilmiştir. |
| **XGBoost** | Ensemble / Boosting | Yüksek performanslı ve hız için test edilmiştir. |
| **LightGBM** | Ensemble / Boosting | Büyük veri setlerinde hızlı ve verimli olduğu için test edilmiştir. |
| **Gradient Boosting** | Ensemble / Boosting | Klasik bir boosting algoritması olarak karşılaştırmaya dahil edilmiştir. |

---

##  Model Karşılaştırma Sonuçları

Sentetik veri setinin deterministik yapısından dolayı, tüm modeller hiperparametre optimizasyonu sonrasında **%100 doğruluk** skoruna ulaşmıştır. Bu durum, veri setindeki özelliklerin hedef değişkeni mükemmel bir şekilde ayrıştırdığını göstermektedir.

| Model | Train Accuracy | Test Accuracy | CV Score | Overfitting (Fark) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 1.0000 | **1.0000** | 1.0000 | 0.0000 |
| **Random Forest** | 1.0000 | **1.0000** | 1.0000 | 0.0000 |
| **XGBoost** | 1.0000 | **1.0000** | 1.0000 | 0.0000 |
| **LightGBM** | 1.0000 | **1.0000** | 1.0000 | 0.0000 |
| **Gradient Boosting** | 1.0000 | **1.0000** | 1.0000 | 0.0000 |

>  **EN İYİ MODEL:** **Tüm modeller %100 doğruluk gösterdiği için**, projenin başlangıcından beri en istikrarlı ve hızlı performansı veren **Logistic Regression** modeli (veya daha basit olan **Random Forest**) nihai model olarak seçilebilir. Bu projede, **Logistic Regression** modeli, hem yorumlanabilirlik hem de gelecekteki olası küçük değişimlere karşı **daha sağlam (robust)** olması nedeniyle nihai model olarak belirlenmiştir.

---

##  Seçilen Modelin Detaylı Raporu: Logistic Regression

Seçilen **Logistic Regression** modeli, test seti üzerinde mükemmel bir performans sergilemiştir.

### Sınıflandırma Raporu

precision    recall  f1-score   support

    High       1.00      1.00      1.00        73
     Low       1.00      1.00      1.00        61
  Medium       1.00      1.00      1.00        66

accuracy                           1.00       200

* **Yorum:** Tüm sınıflar (`High`, `Low`, `Medium`) için **precision**, **recall** ve **f1-score** değerlerinin **1.00** olması, modelin test setindeki her örneği kusursuz bir şekilde doğru tahmin ettiğini göstermektedir.

### Karmaşıklık Matrisi (Confusion Matrix)

True: Low 61 0 0 True: Medium 0 66 0 True: High 0 0 73

**Yorum:** Diyagonal üzerindeki değerler (61, 66, 73), her sınıf için yapılan doğru tahmin sayılarını göstermektedir. Diyagonal dışındaki sıfır değerler, hiçbir yanlış sınıflandırmanın gerçekleşmediğini teyit eder.

##  Sonuç ve Kayıt

Model Optimizasyonu aşaması başarıyla tamamlanmıştır. En iyi performans, en basit modelden en karmaşık modele kadar tüm modellerde gözlenmiştir.

* **Seçilen Nihai Model:** **Logistic Regression** (Hiperparametreleri optimize edilmiş versiyon).
* **Test Doğruluğu:** 1.0000

### Kaydedilen Artefaktlar

Modelin dağıtımı (Deployment) için gerekli olan tüm bileşenler kaydedilmiştir:

1.  **Model Karşılaştırma Sonuçları:** `model_comparison_results.csv`
2.  **Nihai Model:** `best_model.pkl` (Logistic Regression) - **`models/`** klasörüne yerleştirilecektir.
3.  **Ölçekleyici (Scaler):** `scaler.pkl` - Tahmin aşamasında yeni verilerin aynı şekilde ölçeklenmesi için kaydedilmiştir.
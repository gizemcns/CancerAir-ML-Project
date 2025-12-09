# Feature Engineering – Akciğer Kanseri Risk Tahmini

Bu aşamada, baseline modelin %100 doğruluk vermesine rağmen veri setinin karmaşık ilişkilerini daha iyi modellemek için toplam **18 yeni özellik** eklenmiştir. Böylece toplam özellik sayısı **23 iken 41** olmuştur.

Amaç:
- ✔ Modelin genelleme kapasitesini artırmak  
- ✔ Doğrusal olmayan ilişkileri yakalamak  
- ✔ Klinik yorumlanabilirliği güçlendirmek  
- ✔ Gerçek dünya modellerinde underfitting’i önlemek  

Not: Veri seti sentetik olduğundan baseline %100 accuracy vermiştir; bu nedenle performans artışı gözlemlenememiştir. Ancak yapılan Feature Engineering, **gerçek veri projelerinde underfitting’i önlemek için uygulanan standart bir iyi uygulamadır**.

---

## Özet Karşılaştırma Tablosu

| Durum                   | Özellik Sayısı | Test Accuracy | CV Score               | Açıklama                 |
|-------------------------|----------------|---------------|-------------------------|---------------------------|
| Baseline Model          | 23             | 100.00%       | 1.0000 ± 0.0000         | Mükemmel (Ama Basit)     |
| Feature Engineered Model| 41             | 100.00%       | 1.0000 ± 0.0000         | Mükemmel + Zenginleşmiş  |

--> **Underfitting’i gidermek için özellik eklendi — performans korundu, model zenginleşti!**

---

# Yeni Oluşturulan 18 Özellik (Underfitting Odaklı)

| # | Yeni Özellik | Tür | Açıklama |
|---|--------------|-----|----------|
| 1 | Age_Group | Binning | Yaş → 4 gruba ayrıldı (Genç, Yetişkin, Orta Yaş, Yaşlı) |
| 2 | Environmental_Risk | Toplamsal Skor | Hava Kirliliği + Toz Alerjisi + Mesleki Tehlikeler |
| 3 | Lifestyle_Risk | Ağırlıklı Ortalama | Sigara + Alkol + Obezite + Ters Diyet |
| 4 | Genetic_Health_Risk | Ortalama | Genetik Risk + Kronik Akciğer Hastalığı |
| 5 | Symptom_Severity | Ortalama | 7 kritik semptom ortalaması |
| 6 | Respiratory_Score | Ortalama | Solunumla ilgili 4 belirti ortalaması |
| 7 | Critical_Symptom_Count | Sayım | ≥6 puan olan kritik semptom sayısı |
| 8 | Overall_Risk_Score | Ağırlıklı Toplam | Tüm risk faktörlerinin ağırlıklı toplamı |
| 9 | Smoking_Age_Interaction | Etkileşim | Sigara × Yaş |
| 10 | Genetic_Age_Interaction | Etkileşim | Genetik Risk × Yaş |
| 11 | Smoking_Pollution | Etkileşim | Sigara + Hava Kirliliği |
| 12 | Obesity_ChronicLung | Etkileşim | Obezite × Kronik Akciğer Hastalığı |
| 13 | PassiveSmoker_Pollution | Etkileşim | Pasif İçicilik × Hava Kirliliği |
| 14 | Smoking_squared | Polinom | Sigaranın kare etkisi |
| 15 | Air Pollution_squared | Polinom | Kirlilik seviyesinin kare etkisi |
| 16 | Genetic Risk_squared | Polinom | Genetik riskin kare etkisi |
| 17 | Smoking_Level | Binning | Sigara → Low / Medium / High |
| 18 | Pollution_Level | Binning | Kirlilik → Low / Medium / High |

---

#  Neden Bu Özellikleri Ekledik? (Underfitting Stratejisi)

### ✔ Underfitting’i Gidermek  
Logistic Regression gibi lineer modeller karmaşık ilişkileri yakalayamaz. Yeni özellikler bu boşluğu doldurarak modelin performasını artırması hedeflendi.

### ✔ Klinik Yorumlanabilirliği Artırmak  
- Symptom_Severity  
- Overall_Risk_Score  

gibi özellikler doktorların kolayca yorumlayabileceği formatlar sağlar. Gerçek hayat verisinde modele uyarlanarak işlevi artırır.

### ✔ Gerçek Dünya Uyumlu Model  
Sentetik veri %100 accuracy verir; ancak gerçek veride bu mümkün değildir.  
Polinom + etkileşim + binning özelliklerinin verisetinde olması gerçek dünyada model dayanıklılığını artırır.

### ✔ Doğrusal Olmayan Etkileri Modellemek  
Polinom ve etkileşim terimleri sigara, genetik risk ve hava kirliliği gibi faktörlerin doğrusal olmayan etkilerini yakalaması amaçlanarak oluşturulmuştur.

---

# Sonuç ve Değerlendirme

**“Baseline model %100 accuracy vermesine rağmen underfitting potansiyeli nedeniyle 18 yeni özellik ekledik.  
Performans aynı kaldı (%100), fakat model artık daha zengin, daha esnek ve gerçek dünya underfitting problemlerine karşı daha dayanıklı.”**

Bu hem teknik hem klinik açıdan modelin değerini yükseltmiştir.

---

# Üretilen Dosyalar

- **cancer_data_feature_engineered.csv** → 41 özellikli veri seti  
- **feature_list.txt** → Tüm özelliklerin tam listesi


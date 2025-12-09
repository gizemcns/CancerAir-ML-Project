
# Baseline Model Raporu

## 1. Amaç
Bu aşamada hedef, karmaşık algoritmalara geçmeden önce verinin temel ayırt ediciliğini ölçmektir. Bu nedenle baseline model olarak **Logistic Regression** seçilmiştir. Amaç, veri Low / Medium / High risk sınıflarını lineer bir model ile öğrenebiliyor mu, bunu test etmektir.

---

## 2. Baseline Model Sonuçları

### Accuracy
- **Train Accuracy:** %43.38  
- **Test Accuracy:** %27.00  

### 📉 Classification Report (Test)

| Sınıf | Precision | Recall | F1 | Destek |
|-------|-----------|--------|-----|---------|
| 0 (Low)     | 0.23 | 0.15 | 0.18 | 61 |
| 1 (Medium)  | 0.23 | 0.18 | 0.20 | 66 |
| 2 (High)    | 0.31 | 0.45 | 0.37 | 73 |

**Genel Accuracy:** **%27**

---

## 3. Sonuçların Yorumlanması

### 1. Logistic Regression bu veri için yeterli değil
- Veri tamamen kategorik değerlerden oluşmaktadır.
- Kategorilerin ordinal olmadığı halde sayısallaştırılması LR için problem yaratır.
- One-hot encoding sonrası feature sayısı artıyor.
- Lineer model bu karmaşıklığı yakalayamıyor.
- Bu nedenle bu veri seti için uygun değildir.

-> Bu nedenle model sınıfları ayırt edemiyor.  
-> Bu sonuç **normal ve beklenen** bir durumdur.

---

### 2. Model underfitting yapıyor
- Train accuracy düşük (%43)  
- Test accuracy daha düşük (%27)

Model hem eğitim verisini hem test verisini iyi öğrenemiyor → **underfit**.

---

### 3. Veri lineer olarak ayrılabilir değil
Low / Medium / High kategorilerinin sınırları birbirine çok yakındır.  
Bu yüzden LR bu sınıfları ayıracak bir doğrusal sınır çizemiyor.

---

## 4. Baseline Neden Önemli?
Baseline modeli, sonraki aşamalarda daha güçlü modellerin ne kadar gelişme sağladığını ölçmek için referans noktası sağlar.

Bu aşamada elde edilen sonuçlar bize:

- Bu veri setinin **lineer modele uygun olmadığını**,  
- Non-linear modellerin daha iyi performans vereceğini,  
- Özellikle ağaç tabanlı algoritmaların denenmesi gerektiğini göstermiştir.

---

## 5. Çıkarımlar

| Gözlem | Açıklama |
|--------|----------|
| Logistic Regression başarısız | Veri lineer değil, LR karmaşık ilişkileri öğrenemez |
| Tüm değişkenler kategorik | LR kategorik + yüksek kardinaliteli veride zayıf |
| Sınıf ayrımı zayıf | Low/Medium/High kategorileri birbirine çok benzer |
| Tree-based modeller gerekli | Random Forest, XGBoost, CatBoost daha başarılı olacaktır |

---

## 6. Baseline Confusion Matrix

Confusion matrix incelendiğinde modelin sınıfları net şekilde ayıramadığı görülmektedir. Özellikle Medium ve High sınıfları arasında yoğun karışma vardır. Bu da verinin lineer olmayan yapısını göstermektedir.”

![Baseline Confusion Matrix](docs/PNG/baseline_confusion.png)


## 7. Sonraki Adımlar (Final Model)
Baseline analiz sonrası final model olarak şu algoritmalar değerlendirilecektir:

- **Random Forest Classifier**  
- **XGBoost Classifier**  
- **CatBoost Classifier**

Amaç, baseline %27 skorun belirgin şekilde üzerinde bir başarı elde etmektir.

---

## Özet
Baseline modeli doğru bir şekilde referans performansı belirlemiş ve verinin lineer modeller için uygun olmadığını göstermiştir. Final model aşamasında daha güçlü, non-linear algoritmalara geçilecektir. Train Accuracy’nin düşük (%43) ve Test Accuracy’nin daha da düşük (%27) olması modelin underfitting yaptığını göstermektedir. Logistic Regression, tamamen kategorik ve non-lineer yapıdaki bu veri setinde sınıflar arasındaki ilişkileri öğrenememektedir. Bu nedenle baseline model olarak kullanılmış ancak final model için daha güçlü algoritmalara geçilmiştir.

















# 📊 Business Alignment — Kanser Risk Tahmini ML Projesi

Bu doküman, geliştirilen Makine Öğrenmesi tabanlı **Akciğer Kanseri Risk Tahmini** modelinin iş hedefleri, kullanıcı ihtiyaçları ve kurumsal faydalarla nasıl hizalandığını açıklamaktadır.

---

## 🎯 1. İş Problemi ve Bağlam

Akciğer kanseri dünyada en yüksek ölüm oranına sahip hastalıklardan biridir.  
Erken teşhis:

- Tedavi maliyetlerini %60’a kadar düşürür  
- Hayatta kalma oranını ciddi şekilde artırır  
- Sağlık hizmetlerinde yoğunluğu azaltır  

Sağlık kurumları her hastayı ileri tetkiklere yönlendiremez.  
Bu nedenle *ön tarama* amaçlı tahmin modelleri kritik bir ihtiyaçtır.

---

## 🎯 2. Projenin Amacı

Bu projenin amacı:

- Hastanın **risk faktörlerinden** yararlanarak kanser riskini (Low / Medium / High) tahmin etmek  
- Klinik ön değerlendirmeyi hızlandırmak  
- Erken teşhis kapasitesini artırmak  
- Sağlık profesyonellerine destek olmak  

---

## 💼 3. İş Değeri ve Faydalar

### ✔ 1. Maliyet Azaltma
- Gereksiz tarama/test maliyetlerini %30–40 azaltabilir  
- Kaynakların daha verimli kullanılmasını sağlar  

### ✔ 2. Hastane Operasyonel Verimliliği
- Önceliklendirme → yoğun dönemlerde kritik hastaları önce işler  
- Poliklinik yükünü azaltır  

### ✔ 3. Zaman Kazancı
- Doktorun değerlendirmeye ayırdığı süre kısalır  
- İlk karar süreci otomatikleşir  

### ✔ 4. Sağlık Riski Yönetimi
- Sigara, yaşam tarzı, hava kirliliği gibi çevresel risk faktörlerini bütünsel şekilde analiz eder  
- Yüksek riskli bireyleri erken yakalar  

---

## 🧠 4. Neden Makine Öğrenmesi?

- İnsan gözüyle tespit edilemeyen örüntüleri bulur  
- Çok boyutlu risk faktörlerini aynı anda analiz eder  
- Yeni veriler geldikçe kendini günceller  
- “Hangi hastayı önce görmeliyiz?” sorusunu sistematik çözer  

---

## 🩺 5. Kullanıcı Grupları

### 👩‍⚕️ Klinik Personel
Erken uyarı sistemi olarak kullanır.

### 🏥 Hastane Yönetimi
Planlama, maliyet optimizasyonu ve yoğunluk yönetimi için kullanır.

### 👨‍⚕️ Onkologlar
Hangi hasta için ileri tetkik önerileceğini hızlı görür.

### 🧑‍💻 Veri Analitiği Ekipleri
Model performansını izler ve geliştirir.

---

## 📈 6. KPI (Başarı Metrikleri)

Aşağıdaki ölçütler iş başarı göstergesi olarak tanımlanır:

- **Risk tespit doğruluğu**  
- **Yüksek risk hastaları doğru yakalama oranı (Recall)**  
- **Yanlış alarm oranı**  
- **Model inference süresi**  
- **Klinik akışın hızlanması (ör. ön değerlendirme süresi)**  

---

## 🌐 7. Modelin Kullanım Senaryoları

### 1️⃣ **Ön Tarama Sistemi**
Hastaneye başvuran herkeste otomatik risk skoru.

### 2️⃣ **Evde Sağlık / Mobil Uygulama**
“Check-up ön değerlendirme” amaçlı.

### 3️⃣ **Halk Sağlığı Yönetimi**
Belirli bölgelerdeki risk dağılımını inceleme.

### 4️⃣ **Hasta Takibi**
Belirtileri ağırlaşan hastaları erken uyarma.

---

## 🛠️ 8. Teknik Uygulama — İş Değerine Katkı

| Teknik Adım | İş Değeri |
|------------|-----------|
| Feature Engineering | Klinik olarak anlamlı risk skorları üretildi |
| Pipeline | Üretimde sürekli çalışan sistem altyapısı hazırladı |
| API Endpoint | Tüm klinik yazılımlarla entegre olabilir |
| Streamlit UI | Doktorun rahatça kullanabileceği arayüz |
| Model Monitoring | Sistem kalitesini sürekli izleme |

---

## 🔮 9. Gelecek Geliştirmeler

- Gerçek hastane verisi ile yeniden eğitme  
- Radyolojik veri (X-ray, CT) entegrasyonu  
- Domain-expert kalibrasyon modelleri  
- Sensör verileri (hava kalitesi, IoT astım cihazları)  
- Federated learning ile gizlilik uyumlu eğitim  

---

## 🏁 10. Sonuç

Bu proje sadece bir makine öğrenmesi modeli değil,  
**klinik karar verme süreçlerini destekleyen üretim seviyesinde bir çözüm**dür.

Erken teşhis, düşük maliyet ve insan hayatını doğrudan etkileyen bir iş alanında kullanılabilirliği çok yüksektir.


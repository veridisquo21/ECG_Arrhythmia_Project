# 7Hertz: ECG Signal Processing and Arrhythmia Classification
# 7Hertz: EKG Sinyal İşleme ve Aritmi Sınıflandırma

---

## 🇬🇧 English

### 📌 Project Overview
Developed by the **7Hertz** engineering group, this project aims to process raw ECG (Electrocardiogram) signals to enhance clinical interpretability and perform automated arrhythmia classification. The workflow includes digital signal processing (DSP) to remove noise and baseline wander, followed by AI-based classification models.

### ✨ Key Features
* **Signal Pre-processing:** Implementation of a 3rd-order Butterworth High-pass filter (0.5 Hz cutoff) to eliminate low-frequency drifts and respiratory interference.
* **Visualization:** Comparative plotting of raw and filtered signals for data verification.
* **Arrhythmia Detection:** (In Development) Leveraging Deep Learning (CNN/RNN) and DSP algorithms for real-time arrhythmia classification.

### 🛠 Tech Stack
* **Language:** Python 3.x
* **Core Libraries:** `scipy.signal`, `numpy`, `matplotlib`
* **Version Control:** Git & GitHub

---

## 🇹🇷 Türkçe

### 📌 Proje Hakkında
**7Hertz** mühendislik grubu tarafından geliştirilen bu proje, ham EKG (Elektrokardiyogram) sinyallerini işleyerek klinik anlamda yorumlanabilirliği artırmayı ve otomatik aritmi sınıflandırması yapmayı hedeflemektedir. İş akışı, sinyaldeki gürültü ve temel çizgi kaymalarını temizlemek için dijital sinyal işleme (DSP) tekniklerini ve ardından yapay zeka tabanlı sınıflandırma modellerini içerir.

### ✨ Temel Özellikler
* **Sinyal Ön İşleme:** Düşük frekanslı kaymaları ve solunum kaynaklı gürültüleri gidermek için 3. derece Butterworth Yüksek Geçiren filtre (0.5 Hz kesim frekansı) uygulaması.
* **Görselleştirme:** Veri doğrulaması için ham ve filtrelenmiş sinyallerin karşılaştırmalı grafik analizi.
* **Aritmi Tespiti:** (Geliştirme Aşamasında) Derin Öğrenme (CNN/RNN) ve DSP algoritmaları kullanılarak otomatik aritmi sınıflandırması.

### 🛠 Kullanılan Teknolojiler
* **Dil:** Python 3.x
* **Temel Kütüphaneler:** `scipy.signal`, `numpy`, `matplotlib`
* **Versiyon Kontrolü:** Git & GitHub

---

## 🚀 Installation & Usage / Kurulum ve Kullanım

1. **Clone the repository / Depoyu klonlayın:**
   ```bash
   git clone [https://github.com/huseyinayranci/ECG_Arrhythmia_Project.git](https://github.com/huseyinayranci/ECG_Arrhythmia_Project.git)
   cd ECG_Arrhythmia_Project

2. **Install requirements/Gereksinimleri yükleyin:**
    pip install numpy scipy matplotlib

3. **Run the project / Projeyi çalıştırın:**
    python main.py

## 📂 Project Structure / Proje Yapısı

ECG_Arrhythmia_Project/
│
├── src/
│   ├── filters.py         # Filtering functions / Filtreleme fonksiyonları
│   ├── visualization.py   # Plotting tools / Görselleştirme araçları
│   └── main.py            # Main entry point / Ana giriş noktası
│
├── data/                  # ECG Datasets / EKG Veri setleri
├── docs/                  # Project reports / Proje dökümanları
└── README.md              # Project documentation / Tanıtım dosyası

## 👥 Team / Ekip: 7Hertz
Hüseyin Ayrancı - Founder & Team Leader / Kurucu ve Takım Lideri

Yiğit Kürşat Atay - Member & Developer / Üye & Geliştirici

Emir Açıkalın - Member & Developer / Üye & Geliştirici

Muhmmet Melih Kazancı - Member & Developer / Üye & Geliştirici

Erenay Gündoğan - Member & Developer / Üye & Geliştirici

Furkan Uyanık - Member & Developer / Üye & Geliştirici

Ahmet Faruk Şen - Member & Developer / Üye & Geliştirici

## 📄 License / Lisans
This project is licensed under the MIT License. / Bu proje MIT Lisansı ile lisanslanmıştır.
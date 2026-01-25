
---

# ECG Signal Processing and AI-Based Arrhythmia Classification

# EKG Sinyal İşleme ve Yapay Zeka Tabanlı Aritmi Sınıflandırma

---

## 🇬🇧 English

### 📌 Project Overview

This project focuses on the processing of raw ECG (Electrocardiogram) signals to enhance clinical interpretability and perform automated arrhythmia classification using Deep Learning. I developed a comprehensive pipeline that integrates digital signal processing (DSP) to eliminate noise and a hybrid **CNN-LSTM** architecture to achieve high-accuracy classification on the MIT-BIH dataset.

### ✨ Key Features

* **Signal Pre-processing:** Implementation of a 4th-order Butterworth Band-pass filter (0.5 - 45 Hz) to eliminate baseline wander and high-frequency interference.
* **Hybrid Model:** A custom Deep Learning architecture combining **CNN** for morphological feature extraction and **LSTM** for temporal pattern recognition.
* **Loss Function Optimization:** Utilization of **Focal Loss** 

 to address class imbalance and improve the detection of rare arrhythmias.
* **Real-time Dashboard:** A multi-channel monitoring simulation that visualizes signals, calculates BPM, and provides real-time diagnostic predictions.

### 🛠 Tech Stack

* **Language:** Python 3.x
* **AI/ML:** `TensorFlow`, `Keras`, `Scikit-learn`
* **Signal Processing:** `Scipy.signal`, `Numpy`
* **Visualization:** `Matplotlib` (Real-time Animation)

---

## 🇹🇷 Türkçe

### 📌 Proje Hakkında

Bu proje, ham EKG (Elektrokardiyogram) sinyallerini işleyerek klinik anlamda yorumlanabilirliği artırmayı ve Derin Öğrenme kullanarak otomatik aritmi sınıflandırması yapmayı hedeflemektedir. Proje kapsamında, gürültüleri temizlemek için dijital sinyal işleme (DSP) tekniklerini ve MIT-BIH veri seti üzerinde yüksek doğrulukla çalışan hibrit bir **CNN-LSTM** mimarisini içeren uçtan uca bir iş akışı geliştirdim.

### ✨ Temel Özellikler

* **Sinyal Ön İşleme:** Temel çizgi kaymalarını ve yüksek frekanslı gürültüleri gidermek için 4. derece Butterworth Bant Geçiren filtre (0.5 - 45 Hz) uygulaması.
* **Hibrit Model:** Morfolojik özellikleri yakalamak için **CNN**, zamansal ritimleri öğrenmek için **LSTM** katmanlarını birleştiren özgün mimari.
* **Kayıp Fonksiyonu Optimizasyonu:** Sınıf dengesizliğini gidermek ve nadir aritmileri yakalamak için **Focal Loss** 

 kullanımı.
* **Gerçek Zamanlı Panel:** Sinyalleri görselleştiren, anlık BPM hesaplayan ve teşhis koyan çok kanallı profesyonel simülasyon arayüzü.

### 🛠 Kullanılan Teknolojiler

* **Dil:** Python 3.x
* **Yapay Zeka:** `TensorFlow`, `Keras`, `Scikit-learn`
* **Sinyal İşleme:** `Scipy.signal`, `Numpy`
* **Görselleştirme:** `Matplotlib` (Gerçek Zamanlı Animasyon)

---

## 🚀 Installation & Usage / Kurulum ve Kullanım

1. **Clone the repository / Depoyu klonlayın:**
```bash
git clone https://github.com/huseyinayranci/ECG_Arrhythmia_Project.git
cd ECG_Arrhythmia_Project

```


2. **Install requirements / Gereksinimleri yükleyin:**
```bash
pip install numpy scipy matplotlib tensorflow scikit-learn

```


3. **Run the project / Projeyi çalıştırın:**
```bash
python advanced_monitor.py

```



## 📂 Project Structure / Proje Yapısı

ECG_Arrhythmia_Project/
│
├── models/                # Trained model files / Eğitilmiş model dosyaları
├── data/                  # ECG Datasets / EKG Veri setleri
├── src/
│   ├── train_model.py     # Training script / Eğitim kodu
│   └── advanced_monitor.py # Real-time dashboard / Gerçek zamanlı panel
│
├── reports/               # Diagnostic reports / Teşhis raporları
└── EKG_Analiz_Kayitlari.csv # Case logs / Vaka kayıtları

## 👤 Author / Geliştirici:

**Hüseyin Ayrancı** - Electrical and Electronics Engineering Student at Eskişehir Osmangazi University

## 📄 License / Lisans

This project is licensed under the MIT License. / Bu proje MIT Lisansı ile lisanslanmıştır.

---

import matplotlib.pyplot as plt
import numpy as np
import os
from src.data_loader import load_mit_bih_record, load_annotations
from src.preprocess import apply_filters, detect_r_peaks, segment_beats

def main():
    # 1. Veri Yolu Ayarı (Dosya yolunun doğruluğunu kontrol et)
    record_path = 'data/mitdb/100'
    
    if not os.path.exists(record_path + ".hea"):
        print(f"HATA: {record_path}.hea bulunamadı! Lütfen 'data/mitdb/' klasörünü kontrol et.")
        return

    # 2. Veriyi ve Doktor Etiketlerini Yükle
    print("Veriler yükleniyor...")
    signal, fields = load_mit_bih_record(record_path)
    ann_samples, ann_symbols = load_annotations(record_path)
    
    raw_signal = signal[:, 0]  # MLII Kanalı
    fs = fields['fs']          # 360 Hz
    
    # 3. Sinyal İşleme (Filtreleme)
    print("Filtreler uygulanıyor...")
    filtered_signal = apply_filters(raw_signal, fs)
    
    # 4. Kalp Atışı Analizi (R-Peak ve BPM)
    print("Kalp atışları analiz ediliyor...")
    peaks = detect_r_peaks(filtered_signal, fs)
    
    # BPM Hesaplama
    rr_intervals = np.diff(peaks) / fs
    bpm = 60 / np.mean(rr_intervals)
    print(f"--- ANALİZ SONUCU ---")
    print(f"Tespit Edilen Kalp Atış Hızı: {bpm:.2f} BPM")
    print(f"Toplam Dilimlenen Atış Sayısı: {len(peaks)}")
    print(f"----------------------")

    # 5. Segmentasyon (Yapay Zeka İçin Atışları Dilimle)
    # R tepesini merkez alıp sağdan ve soldan 150 örnek (toplam 300) alıyoruz.
    beats = segment_beats(filtered_signal, peaks, window_size=150)

    # 6. Görselleştirme
    plt.figure(figsize=(15, 10))

    # Üst Grafik: Genel Filtrelenmiş Sinyal ve R Tepeleri
    plt.subplot(2, 1, 1)
    plot_range = 2000 # İlk 2000 örneği göster
    plt.plot(filtered_signal[:plot_range], label='Filtrelenmiş Sinyal', color='blue', linewidth=1)
    
    # Sadece görünür aralıktaki tepeleri işaretle
    visible_peaks = peaks[peaks < plot_range]
    plt.plot(visible_peaks, filtered_signal[visible_peaks], "ro", label='Tespit Edilen R Tepeleri')
    
    plt.title(f'EKG Sinyal İşleme ve R-Peak Tespiti (BPM: {bpm:.1f})')
    plt.ylabel('Genlik')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Alt Grafik: Dilimlenmiş İlk 5 Kalp Atışı (Model Girişi Örneği)
    for i in range(5):
        if i < len(beats):
            plt.subplot(2, 5, i + 6)
            plt.plot(beats[i], color='green')
            plt.title(f"Atış {i+1}")
            plt.grid(True, alpha=0.2)
            if i == 0: plt.ylabel('Segmentler')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
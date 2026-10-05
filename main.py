import matplotlib.pyplot as plt
import numpy as np
import os
from src.data_loader import (
    AAMI_CLASSES,
    load_mit_bih_record,
    load_annotations,
    select_signal_channel,
)
from src.preprocess import (
    apply_filters,
    calculate_bpm,
    detect_r_peaks,
    peak_detection_metrics,
    segment_beats,
)

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
    reference_samples = np.asarray(
        [sample for sample, symbol in zip(ann_samples, ann_symbols) if symbol in AAMI_CLASSES]
    )
    
    raw_signal = select_signal_channel(signal, fields["sig_name"])
    fs = fields['fs']          # 360 Hz
    
    # 3. Sinyal İşleme (Filtreleme)
    print("Filtreler uygulanıyor...")
    filtered_signal = apply_filters(raw_signal, fs)
    
    # 4. Kalp Atışı Analizi (R-Peak ve BPM)
    print("Kalp atışları analiz ediliyor...")
    peaks = detect_r_peaks(filtered_signal, fs)
    
    # BPM Hesaplama
    bpm = calculate_bpm(peaks, fs)
    metrics = peak_detection_metrics(peaks, reference_samples, fs)
    print(f"--- ANALİZ SONUCU ---")
    print(f"Tespit Edilen Kalp Atış Hızı: {bpm:.2f} BPM" if bpm else "BPM: yeterli R tepesi yok")
    print(f"Tespit Edilen R Tepesi Sayısı: {len(peaks)}")
    print(f"R-peak sensitivity: {metrics['sensitivity']:.3f}")
    print(f"R-peak PPV: {metrics['ppv']:.3f}")
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
    
    bpm_title = f"{bpm:.1f}" if bpm else "N/A"
    plt.title(f'EKG Sinyal İşleme ve R-Peak Tespiti (BPM: {bpm_title})')
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
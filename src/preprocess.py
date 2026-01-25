import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def apply_filters(data, fs=360):
    """
    EKG sinyaline sırasıyla High-pass, Notch ve Low-pass filtre uygular.
    fs=360: MIT-BIH veri setinin standart örnekleme frekansıdır.
    """
    # 1. Baseline Wander Giderme (High-pass: 0.5 Hz)
    # Nefes alıp verme gibi düşük frekanslı kaymaları temizler.
    nyq = 0.5 * fs
    b_high, a_high = signal.butter(3, 0.5 / nyq, btype='high')
    data_filtered = signal.filtfilt(b_high, a_high, data)

    # 2. Şebeke Gürültüsü Giderme (Notch: 50 Hz)
    # Türkiye'deki şehir şebekesinin 50Hz frekansını keser.
    b_notch, a_notch = signal.iirnotch(50.0 / nyq, 30.0)
    data_filtered = signal.filtfilt(b_notch, a_notch, data_filtered)

    # 3. Yüksek Frekans Gürültüsü Giderme (Low-pass: 45 Hz)
    # Kas titremeleri (EMG) ve cihaz gürültülerini temizler.
    b_low, a_low = signal.butter(4, 45.0 / nyq, btype='low')
    data_filtered = signal.filtfilt(b_low, a_low, data_filtered)

    return data_filtered

def plot_results(original, filtered, title="EKG Filtreleme Sonucu"):
    """Sonuçları karşılaştırmalı olarak çizer."""
    plt.figure(figsize=(15, 5))
    plt.plot(original[:1500], label='Ham Sinyal', color='gray', alpha=0.5)
    plt.plot(filtered[:1500], label='Temizlenmiş Sinyal', color='blue')
    plt.title(title)
    plt.xlabel('Örnek Sayısı (n)')
    plt.ylabel('Genlik (mV)')
    plt.legend()
    plt.grid(True)
    plt.show()
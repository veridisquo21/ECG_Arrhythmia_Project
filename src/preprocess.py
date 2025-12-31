import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def butter_highpass(cutoff, fs, order=5):
    """Yüksek geçiren filtre katsayılarını hesaplar."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def filter_baseline_wander(data, fs, cutoff=0.5):
    """0.5 Hz altındaki düşük frekanslı kaymaları temizler."""
    b, a = butter_highpass(cutoff, fs, order=3)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

# Test için görselleştirme (Opsiyonel)
def plot_comparison(original, filtered):
    plt.figure(figsize=(12, 6))
    plt.plot(original[:2000], label='Ham Sinyal', alpha=0.5)
    plt.plot(filtered[:2000], label='Filtrelenmiş Sinyal (High-pass 0.5Hz)', color='red')
    plt.title('Baseline Wander Temizleme Sonucu')
    plt.legend()
    plt.grid(True)
    plt.show()
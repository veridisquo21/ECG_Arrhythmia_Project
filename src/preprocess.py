import numpy as np
from scipy.signal import butter, lfilter, iirnotch, find_peaks

def apply_filters(signal, fs):
    """
    EKG sinyaline High-pass, Low-pass ve Notch filtre uygular.
    """
    # 1. High-pass Filter (0.5 Hz) - Baseline kaymasını engeller
    nyq = 0.5 * fs
    low = 0.5 / nyq
    b, a = butter(1, low, btype='high')
    signal = lfilter(b, a, signal)

    # 2. Low-pass Filter (40 Hz) - Kas gürültüsünü engeller
    high = 40 / nyq
    b, a = butter(4, high, btype='low')
    signal = lfilter(b, a, signal)

    # 3. Notch Filter (50 Hz) - Şebeke gürültüsünü engeller
    q = 30.0
    freq = 50.0
    b, a = iirnotch(freq, q, fs)
    signal = lfilter(b, a, signal)

    return signal

def detect_r_peaks(filtered_signal, fs):
    """
    Filtrelenmiş sinyaldeki R tepelerini tespit eder.
    """
    # Mesafe ve yükseklik parametreleri record 100 için optimize edilmiştir
    distance = int(0.6 * fs) 
    height = np.mean(filtered_signal) + 0.5 * np.std(filtered_signal)
    
    peaks, _ = find_peaks(filtered_signal, distance=distance, height=height)
    return peaks

def segment_beats(signal, peaks, window_size=150):
    """
    R tepelerini merkez alarak sinyali küçük parçalara (segmentlere) böler.
    window_size: Tepenin sağından ve solundan kaç örnek alınacağı.
    """
    beats = []
    for peak in peaks:
        start = peak - window_size
        end = peak + window_size
        
        # Sinyal sınırlarını kontrol et
        if start > 0 and end < len(signal):
            beat = signal[start:end]
            beats.append(beat)
            
    return np.array(beats)

def normalize_beat(beat):
    """
    Sinyali [0, 1] arasına çeker (Min-Max Scaling).
    """
    minimum = np.min(beat)
    maximum = np.max(beat)
    if maximum - minimum == 0:
        return beat
    return (beat - minimum) / (maximum - minimum)
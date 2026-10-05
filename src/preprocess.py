import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

def apply_filters(signal, fs, lowcut=0.5, highcut=40.0):
    """Apply one zero-phase Butterworth band-pass filter to an offline record."""
    if highcut >= fs / 2:
        raise ValueError(f"highcut must be below Nyquist frequency ({fs / 2:g} Hz)")
    sos = butter(4, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=float))

def detect_r_peaks(filtered_signal, fs):
    """
    Filtrelenmiş sinyaldeki R tepelerini tespit eder.
    """
    distance = max(1, int(0.22 * fs))
    height = np.mean(filtered_signal) + 0.8 * np.std(filtered_signal)
    
    peaks, _ = find_peaks(filtered_signal, distance=distance, height=height)
    return peaks


def calculate_bpm(peaks, fs, min_peaks=2):
    """Calculate BPM from consecutive R peaks, or return None if insufficient."""
    if len(peaks) < min_peaks:
        return None
    intervals = np.diff(peaks) / float(fs)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if not len(intervals):
        return None
    median = np.median(intervals)
    valid = intervals[(intervals >= 0.5 * median) & (intervals <= 1.5 * median)]
    return float(60.0 / np.mean(valid)) if len(valid) else None


def peak_detection_metrics(detected, reference, fs, tolerance_seconds=0.15):
    """Return sensitivity and PPV for peaks matched within a timing tolerance."""
    detected = np.asarray(detected)
    reference = np.asarray(reference)
    tolerance = tolerance_seconds * fs
    matched = set()
    true_positives = 0
    for peak in detected:
        candidates = np.flatnonzero(np.abs(reference - peak) <= tolerance)
        candidates = [int(i) for i in candidates if int(i) not in matched]
        if candidates:
            matched.add(candidates[0])
            true_positives += 1
    sensitivity = true_positives / len(reference) if len(reference) else 0.0
    precision = true_positives / len(detected) if len(detected) else 0.0
    return {"sensitivity": sensitivity, "ppv": precision, "true_positives": true_positives}

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
        if start >= 0 and end <= len(signal):
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
        return np.zeros_like(beat, dtype=float)
    return (beat - minimum) / (maximum - minimum)
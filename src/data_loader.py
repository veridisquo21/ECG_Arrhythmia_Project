import wfdb
import os

def load_mit_bih_record(record_path):
    """
    MIT-BIH veri setinden bir kaydı yükler.
    """
    record = wfdb.rdrecord(record_path)
    # Sinyal verisi ve metadata (fs, kanal adları vb.) döner
    return record.p_signal, {'fs': record.fs, 'sig_name': record.sig_name}

def load_annotations(record_path):
    """
    Kayıt altındaki doktor etiketlerini (annotation) ve zamanlarını döndürür.
    """
    # .atr uzantılı dosyayı okur
    annotation = wfdb.rdann(record_path, 'atr')
    
    # num_samples: Her bir etiketin sinyaldeki konumu (index)
    # symbols: Her bir etiketin türü (N, V, A, L, R vb.)
    return annotation.sample, annotation.symbol

# AAMI Sınıflandırma Sözlüğü
AAMI_CLASSES = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,      # N: Normal
    'A': 1, 'a': 1, 'S': 1, 'J': 1,              # S: Supraventricular
    'V': 2, 'E': 2,                              # V: Ventricular
    'F': 3,                                      # F: Fusion
    '/': 4, 'f': 4, 'Q': 4                       # Q: Unknown / Paced
}
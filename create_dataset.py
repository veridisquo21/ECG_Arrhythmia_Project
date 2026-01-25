import numpy as np
import os
from src.data_loader import load_mit_bih_record, load_annotations, AAMI_CLASSES
from src.preprocess import apply_filters, segment_beats, normalize_beat

def create_dataset():
    data_dir = 'data/mitdb'
    # Klasördeki tüm record numaralarını al (Örn: 100, 101...)
    records = [f.replace('.hea', '') for f in os.listdir(data_dir) if f.endswith('.hea')]
    
    X = [] # Sinyal dilimleri
    y = [] # Etiketler (0, 1, 2, 3, 4)

    print(f"{len(records)} kayıt işleniyor...")

    for r in records:
        path = os.path.join(data_dir, r)
        try:
            # 1. Yükle
            signal, fields = load_mit_bih_record(path)
            ann_samples, ann_symbols = load_annotations(path)
            
            # 2. Filtrele
            filtered = apply_filters(signal[:, 0], fields['fs'])
            
            # 3. Dilimle ve Normalize Et
            for i in range(len(ann_samples)):
                label = AAMI_CLASSES.get(ann_symbols[i], None)
                
                # Sadece AAMI sınıflarına giren ve sinyal sınırlarında olan atışları al
                if label is not None:
                    start = ann_samples[i] - 150
                    end = ann_samples[i] + 150
                    if start > 0 and end < len(filtered):
                        beat = filtered[start:end]
                        beat = normalize_beat(beat) # [0, 1] arası
                        X.append(beat)
                        y.append(label)
        except Exception as e:
            print(f"Kayıt {r} işlenirken hata oluştu: {e}")

    # NumPy dizilerine çevir
    X = np.array(X)
    y = np.array(y)

    # Veriyi Kaydet (AI eğitimi için hazır)
    np.savez('data/processed_dataset.npz', X=X, y=y)
    print(f"İşlem tamamlandı! Toplam {len(X)} atış kaydedildi.")

if __name__ == "__main__":
    create_dataset()
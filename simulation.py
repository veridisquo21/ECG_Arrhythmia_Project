import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Modeli ve Test Verisini Yükle
# En son eğittiğimiz v3_final veya focal_loss modelini kullanıyoruz
model = tf.keras.models.load_model('models/ecg_model_v3_final.keras')
data = np.load('data/processed_dataset.npz')
X_test, y_test = data['X'], data['y']

# Sınıf isimleri (MIT-BIH standartlarına göre)
class_map = {0: "Normal", 1: "Supraventricular", 2: "Ventricular", 3: "Fusion", 4: "Unknown"}

# Simülasyon için rastgele bir sinyal seçelim
sample_idx = np.random.randint(0, len(X_test))
ecg_signal = X_test[sample_idx]
true_label = class_map[int(y_test[sample_idx])]

# 2. Görselleştirme Hazırlığı
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot([], [], lw=2, color='#2ecc71')
ax.set_ylim(-1, 1.5) # EKG genliğine göre ayarla
ax.set_xlim(0, 300)
ax.set_title(f"  Gerçek Zamanlı İzleme - Gerçek Teşhis: {true_label}")
ax.set_xlabel("Zaman (Örnek)")
ax.set_ylabel("Genlik (mV)")

# Tahmin metni alanı
prediction_text = ax.text(10, 1.2, '', fontsize=12, fontweight='bold', color='red')

# 3. Simülasyon Güncelleme Fonksiyonu
def init():
    line.set_data([], [])
    prediction_text.set_text('')
    return line, prediction_text

def update(frame):
    # Kayan pencere simülasyonu (0'dan frame'e kadar olan veriyi göster)
    x = np.arange(0, frame)
    y = ecg_signal[:frame]
    
    line.set_data(x, y)
    
    # Pencere 300 örneğe ulaştığında modelden tahmin iste
    if frame == 300:
        # Modeli beslemek için boyutu düzenle (1, 300, 1)
        input_data = ecg_signal.reshape(1, 300, 1)
        pred_probs = model.predict(input_data, verbose=0)
        pred_idx = np.argmax(pred_probs)
        confidence = np.max(pred_probs) * 100
        
        pred_label = class_map[pred_idx]
        prediction_text.set_text(f"MODEL TEŞHİSİ: {pred_label} (%{confidence:.1f})")
        
        # Eğer aritmi tespit edilirse rengi değiştir
        if pred_idx != 0:
            line.set_color('#e74c3c') # Kırmızı
        else:
            line.set_color('#2ecc71') # Yeşil

    return line, prediction_text

# 4. Animasyonu Başlat
# frames=301 (verinin tamamı), interval=20 (ms cinsinden hız)
ani = FuncAnimation(fig, update, frames=len(ecg_signal)+1,
                    init_func=init, blit=True, repeat=False, interval=10)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
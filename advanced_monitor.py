import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt
import time
import csv
import os

# --- 1. DSP ve Teşhis Hazırlığı ---
def butter_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=125, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def get_realistic_bpm(class_idx):
    if class_idx == 0: return np.random.randint(60, 85)
    if class_idx == 1: return np.random.randint(90, 115)
    if class_idx == 2: return np.random.randint(110, 155)
    return np.random.randint(70, 95)

# Model ve Veri Yükleme
model = tf.keras.models.load_model('models/ecg_model_v3_final.keras')
data = np.load('data/processed_dataset.npz')
X_test, y_test = data['X'], data['y']
class_map = {0: "Normal", 1: "Supraventricular", 2: "Ventricular", 3: "Fusion", 4: "Unknown"}

# --- 2. Vaka Günlüğü ---
log_file = "Vaka_Kayitlari.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Tarih_Saat', 'Kanal', 'Hasta_ID', 'BPM', 'Tespit_Edilen', 'Guven_Orani'])

# --- 3. Dashboard Kurulumu ---
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=100)
fig.suptitle("Dinamik ve Rastgele EKG Analiz Sistemi", fontsize=18, color='#3498db')
axes = axes.flatten()

num_channels = 4
filtered_signals, patient_ids, current_classes = [], [], []

def refresh_patients():
    """Tamamen rastgele ve çeşitli sinyaller seçer."""
    global filtered_signals, patient_ids, current_classes
    filtered_signals, patient_ids, current_classes = [], [], []
    
    # Tüm test seti içinden 4 tane tamamen rastgele ve benzersiz indeks seç
    # 'replace=False' her kanalın farklı bir hasta olmasını sağlar
    random_idxs = np.random.choice(len(X_test), num_channels, replace=False)
    
    for idx in random_idxs:
        raw = X_test[idx].flatten()
        clean = butter_bandpass_filter(raw)

        # Rastgele Kaydırma (Atım yerini değiştirme)
        clean = np.roll(clean, np.random.randint(-100, 100))

        # Baz Hattı Dalgalanması (Nefes alma simülasyonu)
        t = np.linspace(0, 1, 300)
        clean = clean + 0.04 * np.sin(2 * np.pi * 0.5 * t + np.random.rand())

        # Genlik Normalizasyonu ve Ölçekleme
        scale = np.random.uniform(0.7, 1.1)
        clean = (clean - np.min(clean)) / (np.max(clean) - np.min(clean) + 1e-7)
        clean = clean * scale

        filtered_signals.append(clean)
        patient_ids.append(idx)
        current_classes.append(y_test[idx])

refresh_patients()

lines, texts, bpm_texts = [], [], []
colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f'] # Her kanal farklı renk

for i in range(num_channels):
    ax = axes[i]
    line, = ax.plot([], [], lw=2, color=colors[i])
    lines.append(line)
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.1, 1.3)
    ax.set_title(f"MONİTÖR {i+1} | Hasta ID: #{patient_ids[i]}", fontsize=10)
    ax.grid(True, alpha=0.05)
    texts.append(ax.text(5, 1.15, 'Analiz Başlıyor...', fontsize=9, fontweight='bold', color='#f1c40f'))
    bpm_texts.append(ax.text(230, 1.15, 'BPM: --', fontsize=11, color='#3498db', fontweight='bold'))

# --- 4. Animasyon Döngüsü ---
waiting_frames = 0

def update(frame):
    global waiting_frames
    if waiting_frames > 0:
        waiting_frames -= 1
        return lines + texts + bpm_texts

    if frame == 300:
        log_entries = []
        for i in range(num_channels):
            # Model tahmini
            pred = model.predict(filtered_signals[i].reshape(1, 300, 1), verbose=0)
            p_idx = np.argmax(pred)
            diag = class_map[p_idx]
            conf = np.max(pred) * 100
            
            # Gerçek BPM simülasyonu
            bpm = get_realistic_bpm(p_idx)
            
            texts[i].set_text(f"TEŞHİS: {diag} (%{conf:.1f})")
            bpm_texts[i].set_text(f"BPM: {bpm}")
            
            # Alarm durumu (Normal değilse kırmızı yap)
            if p_idx != 0:
                lines[i].set_color('#e74c3c')
                texts[i].set_color('#e74c3c')
            
            log_entries.append([time.strftime("%Y-%m-%d %H:%M:%S"), i+1, patient_ids[i], bpm, diag, f"{conf:.2f}%"])

        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(log_entries)
        
        waiting_frames = 60 # Analiz sonrası 2 saniye bekle
        refresh_patients()
        for i in range(num_channels):
            axes[i].set_title(f"MONİTÖR {i+1} | Hasta ID: #{patient_ids[i]}", fontsize=10)
            lines[i].set_color(colors[i]) # Renkleri orijinaline döndür
            texts[i].set_color('#f1c40f')
        return lines + texts + bpm_texts

    for i in range(num_channels):
        lines[i].set_data(np.arange(frame), filtered_signals[i][:frame])
        # BPM'i akışta göster
        if frame > 150:
            bpm_texts[i].set_text(f"BPM: {get_realistic_bpm(current_classes[i])}")

    return lines + texts + bpm_texts

ani = FuncAnimation(fig, update, frames=301, blit=False, repeat=True, interval=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
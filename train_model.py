import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 1. Veri Yükleme ve Augmentation (Gürültü Ekleme)
data = np.load('data/processed_dataset.npz')
X, y = data['X'], data['y']
X = X.reshape(X.shape[0], X.shape[1], 1)

def add_noise(signal, noise_level=0.01):
    """Sinyale rastgele gürültü ekleyerek veriyi çeşitlendirir."""
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

# Sadece azınlık sınıflarını (1 ve 2) gürültü ekleyerek çoğaltalım
unique, counts = np.unique(y, return_counts=True)
X_list, y_list = [X], [y]

for cls in [1, 2]:
    idx = np.where(y == cls)[0]
    if len(idx) > 0:
        # Mevcut veriyi al ve üzerine gürültü ekleyerek 10 katına çıkar
        for _ in range(10):
            X_aug = add_noise(X[idx])
            X_list.append(X_aug)
            y_list.append(y[idx])

X_final = np.concatenate(X_list, axis=0)
y_final = np.concatenate(y_list, axis=0)

# 2. Bölme ve Dengeleme
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)

# --- [YENİ: FOCAL LOSS TANIMLAMASI] ---
# Modelin "kolay" örnekleri (Normal) hızlı geçip "zor" örneklere odaklanmasını sağlar.
def focal_loss(gamma=2., alpha=4.):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=3) # Sınıf sayınıza göre ayarlayın
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(K.sum(loss, axis=-1))
    return focal_loss_fixed

# 3. Hibrit Model (CNN + LSTM)
model = models.Sequential([
    layers.Conv1D(32, 7, padding='same', input_shape=(X.shape[1], 1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling1D(2),
    
    layers.Conv1D(64, 5, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.GlobalAveragePooling1D(),
    layers.Reshape((64, 1)), # LSTM girişi için boyut düzenleme
    
    layers.LSTM(32),
    layers.Dropout(0.4),
    layers.Dense(3, activation='softmax')
])

# 4. Derleme ve Eğitim
# Öğrenme hızını orta bir değere (5e-5) çekiyoruz
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
              loss=focal_loss(gamma=2.0, alpha=4.0),
              metrics=['accuracy'])

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print("  Focal-Learning Başlıyor...")
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    validation_data=(X_test, y_test), 
                    batch_size=32,
                    callbacks=[early_stop])

# 5. Değerlendirme
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('  Focal Loss Performansı')
plt.show()
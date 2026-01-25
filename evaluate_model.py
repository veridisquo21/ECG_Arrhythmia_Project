import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Veriyi ve Modeli Yükle
data = np.load('data/processed_dataset.npz')
X_test, y_test = data['X'], data['y']
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = tf.keras.models.load_model('models/ecg_model_v1.h5')

# 2. Tahminleri Al
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 3. Karmaşıklık Matrisini Çiz
cm = confusion_matrix(y_test, y_pred_classes)
classes = ['Normal', 'Supravent', 'Ventricular', 'Fusion', 'Unknown']

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('  EKG Sınıflandırma Performansı')
plt.show()

# 4. Detaylı Rapor (F1-Score vb.)
print(classification_report(y_test, y_pred_classes, target_names=classes))
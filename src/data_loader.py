import wfdb
import matplotlib.pyplot as plt
import os

def initial_setup():
    """Veri klasörünü kontrol eder ve Record 100'ü indirir."""
    data_path = 'data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    print("Veri seti kontrol ediliyor...")
    # MIT-BIH Arrhythmia Database (mitdb) indiriliyor
    wfdb.dl_database('mitdb', dl_dir=data_path, records=['100'], overwrite=False)
    
    # Kaydı oku (İlk 5 saniye: 360 * 5 = 1800 örnek)
    record = wfdb.rdrecord(os.path.join(data_path, '100'), sampto=1800)
    
    print(f"Kayıt Okundu: {record.record_name}")
    print(f"Sinyal Kanalları: {record.sig_name}")
    
    # Çizim
    plt.figure(figsize=(10, 4))
    plt.plot(record.p_signal[:, 0], label=record.sig_name[0])
    plt.title("Kayıt 100 - İlk 5 Saniye (Ham Sinyal)")
    plt.ylabel("Genlik (mV)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    initial_setup()
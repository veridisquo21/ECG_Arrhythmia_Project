import wfdb
import os

# 1. Klasörü oluştur (Eğer yoksa)
data_dir = 'data/mitdb'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print("MIT-BIH Database indiriliyor, lütfen bekleyin...")

# 2. Sadece ihtiyacımız olan dosyaları (Örn: Record 100) indir
# Tüm veritabanını indirmek istersen dl_dir='data/mitdb' yapabilirsin.
wfdb.dl_database('mitdb', dl_dir=data_dir, records=['100', '101'], keep_log=False)

print(f"İşlem tamam! Dosyalar şuraya kaydedildi: {data_dir}")
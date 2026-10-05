import wfdb
from pathlib import Path

data_dir = Path("data/mitdb")
data_dir.mkdir(parents=True, exist_ok=True)

print("MIT-BIH Database indiriliyor, lütfen bekleyin...")
wfdb.dl_database("mitdb", dl_dir=str(data_dir), records=None, keep_log=False)

print(f"İşlem tamam! Dosyalar şuraya kaydedildi: {data_dir}")
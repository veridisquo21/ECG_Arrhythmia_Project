import wfdb
from pathlib import Path

data_dir = Path("data/mitdb")
data_dir.mkdir(parents=True, exist_ok=True)

print("MIT-BIH Database indiriliyor, lütfen bekleyin...")
records = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
wfdb.dl_database("mitdb", dl_dir=str(data_dir), records=records)

print(f"İşlem tamam! Dosyalar şuraya kaydedildi: {data_dir}")

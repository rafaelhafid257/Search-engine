import json
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- SETUP NLP ---
print("Memulai proses stemming data... (Ini mungkin memakan waktu)")
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()

def process_text(text):
    if not text: return []
    # 1. Lowercase & Regex Cleaning
    clean_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    # 2. Stopword
    clean_text = stopword_remover.remove(clean_text)
    # 3. Stemming (Proses terberat)
    stemmed_text = stemmer.stem(clean_text)
    # 4. Tokenizing
    return stemmed_text.split()

# --- LOAD DATA ASLI ---
with open("pariwisata.json", encoding="utf-8") as f:
    raw_data = json.load(f)

processed_entries = []

# --- LOOP & PROCESS ---
for prov_id, prov_data in raw_data.get('provinsi', {}).items():
    prov_name = prov_data.get("nama", "")
    
    # Proses Objek Wisata
    for item in prov_data.get("objek_pariwisata", []):
        full_content = item.get("nama", "") + " " + item.get("deskripsi", "")
        processed_entries.append({
            "provinsi": prov_name,
            "type": "Wisata",
            "nama": item.get("nama", ""),
            "original_konten": item.get("deskripsi", ""), # Simpan teks asli untuk ditampilkan
            "tokens": process_text(full_content),         # Simpan tokens untuk BM25
            "gambar": item.get("gambar", "")
        })
        
    # Proses Makanan
    for item in prov_data.get("makanan_khas", []):
        full_content = item.get("nama", "") + " " + item.get("deskripsi", "")
        processed_entries.append({
            "provinsi": prov_name,
            "type": "Kuliner",
            "nama": item.get("nama", ""),
            "original_konten": item.get("deskripsi", ""),
            "tokens": process_text(full_content),
            "gambar": item.get("gambar", "")
        })

# --- SIMPAN HASIL KE FILE BARU ---
output_file = "data_siap_pakai.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed_entries, f, ensure_ascii=False, indent=2)

print(f"Selesai! Data tersimpan di '{output_file}'. Sekarang app.py akan ngebut! 🚀")
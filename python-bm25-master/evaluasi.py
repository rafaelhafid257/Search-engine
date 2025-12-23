import json
from bm25 import BM25
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- 1. SETUP SISTEM (Sama seperti app.py) ---
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()
factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()

def preprocess_query(text):
    text = text.lower()
    text = stopword_remover.remove(text)
    return stemmer.stem(text).split()

# Load Data
with open("data_siap_pakai.json", encoding="utf-8") as f:
    doc_entries = json.load(f)
corpus = [doc['tokens'] for doc in doc_entries]
bm25 = BM25(corpus)

# --- 2. DEFINISI GROUND TRUTH (KUNCI JAWABAN) ---
# Format: "Query": ["Kata kunci yang HARUS muncul di judul hasil"]
# Anda bisa menambahkan skenario lain sesuai Bab 3.5
test_scenarios = {
    "wisata danau di sumatera": ["Danau Toba", "Danau Singkarak", "Danau Maninjau", "Danau Ranau"],
    "kuliner pedas bali": ["Betutu", "Lawar", "Sate Lilit"],
    "candi di jawa": ["Borobudur", "Prambanan", "Mendut", "Penataran"],
    "makanan khas palembang": ["Pempek", "Tekwan", "Laksan"],
    "pantai indah": ["Kuta", "Sanur", "Senggigi", "Parai Tenggiri"]
}

# --- 3. JALANKAN PENGUJIAN ---
print(f"{'QUERY':<30} | {'PRECISION':<10} | {'RECALL':<10}")
print("-" * 55)

total_precision = 0
total_recall = 0

for query, expected_keywords in test_scenarios.items():
    # Proses Query
    query_tokens = preprocess_query(query)
    
    # Ambil Top 10 Hasil
    ranked_indices = bm25.ranked(query_tokens, 10)
    retrieved_docs = [doc_entries[idx]['nama'] for idx in ranked_indices]
    
    # Hitung Relevansi (Apakah hasil mengandung kata kunci yang diharapkan?)
    relevant_retrieved = 0
    for doc_name in retrieved_docs:
        # Cek apakah nama dokumen cocok dengan salah satu kunci jawaban
        if any(keyword.lower() in doc_name.lower() for keyword in expected_keywords):
            relevant_retrieved += 1
            
    # --- RUMUS SKRIPSI ---
    # Precision = (Dokumen Relevan Ditemukan) / (Total Dokumen Ditemukan Sistem)
    precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
    
    # Recall = (Dokumen Relevan Ditemukan) / (Total Dokumen Relevan di Kunci Jawaban)
    # Note: Ini estimasi sederhana, idealnya kita tahu total dokumen relevan di SELURUH database
    recall = relevant_retrieved / len(expected_keywords) if expected_keywords else 0
    
    print(f"{query:<30} | {precision:.2f}       | {recall:.2f}")
    
    total_precision += precision
    total_recall += recall

# --- RATA-RATA ---
n = len(test_scenarios)
print("-" * 55)
print(f"RATA-RATA SISTEM               | {total_precision/n:.2f}       | {total_recall/n:.2f}")
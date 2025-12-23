import pandas as pd
import math

# --- 1. DATA SAMPEL ---
documents = [
    {"id": "D1", "text": "Wisata Pantai Kuta di Bali sangat indah"},
    {"id": "D2", "text": "Pantai Sanur menawarkan pemandangan indah matahari terbit"},
    {"id": "D3", "text": "Wisata kuliner di Bali sangat lezat dan murah"},
]
query_raw = "wisata pantai indah di bali"

# Simulasi Preprocessing
docs_processed = [
    {"id": "D1", "tokens": ["wisata", "pantai", "kuta", "bali", "indah"]},
    {"id": "D2", "tokens": ["pantai", "sanur", "tawar", "pandang", "indah", "matahari", "terbit"]},
    {"id": "D3", "tokens": ["wisata", "kuliner", "bali", "lezat", "murah"]},
]
query_processed = ["wisata", "pantai", "indah", "bali"]

# Parameter
k1 = 1.2
b = 0.75
N = len(docs_processed)
avgdl = sum(len(d['tokens']) for d in docs_processed) / N

# --- 2. HITUNG IDF ---
terms_unique = sorted(list(set(t for d in docs_processed for t in d['tokens'])))
idf_dict = {}
for term in terms_unique:
    n_qi = sum(1 for d in docs_processed if term in d['tokens'])
    idf_val = math.log(N - n_qi + 0.5) - math.log(n_qi + 0.5)
    idf_dict[term] = idf_val

# --- 3. HITUNG SKOR & TULIS RUMUS ---
score_rows = []

for doc in docs_processed:
    doc_id = doc['id']
    dl = len(doc['tokens'])
    K = k1 * (1 - b + b * (dl / avgdl))
    
    total_score = 0
    
    # Kita buat baris per KATA KUNCI agar rincian rumusnya terlihat jelas
    for term in query_processed:
        if term not in terms_unique: continue
        
        # Ambil variabel
        tf = doc['tokens'].count(term)
        idf = idf_dict.get(term, 0)
        
        # Hitung Komponen Rumus BM25
        numerator = idf * tf * (k1 + 1)
        denominator = tf + K
        term_score = numerator / denominator
        
        total_score += term_score
        
        # --- INI BAGIAN PENTING: MENULIS TEKS RUMUS ---
        # Format visual: (IDF * TF * (k1+1)) / (TF + K)
        rumus_visual = f"({idf:.3f} * {tf} * {k1+1}) / ({tf} + {K:.3f})"
        
        score_rows.append({
            "Dokumen": doc_id,
            "Kata Kunci": term,
            "TF (Freq)": tf,
            "IDF (Bobot)": round(idf, 4),
            "K (Normalisasi)": round(K, 4),
            "Rincian Rumus (Visual)": rumus_visual, # <--- Ini yang Abang cari
            "Hasil Skor Kata": round(term_score, 4)
        })

# Buat DataFrame
df_rincian = pd.DataFrame(score_rows)

# Hitung Total Skor per Dokumen
df_total = df_rincian.groupby("Dokumen")["Hasil Skor Kata"].sum().reset_index()
df_total.rename(columns={"Hasil Skor Kata": "Total Skor BM25"}, inplace=True)
df_total = df_total.sort_values(by="Total Skor BM25", ascending=False)

# --- 4. EXPORT EXCEL ---
filename = "Perhitungan_BM25_Dengan_Rumus.xlsx"
with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
    df_rincian.to_excel(writer, sheet_name='Rincian Per Kata', index=False)
    df_total.to_excel(writer, sheet_name='Ranking Akhir', index=False)
    
    # Auto-adjust column width (Supaya kolom Rumus terbaca semua)
    worksheet = writer.sheets['Rincian Per Kata']
    worksheet.set_column('F:F', 35) # Lebarkan kolom rumus

print(f"Sukses! Cek file '{filename}'. Sekarang ada kolom rincian rumusnya!")
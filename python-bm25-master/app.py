import json
import re
from flask import Flask, render_template, request
from markupsafe import Markup
from bm25 import BM25
from collections import defaultdict

# Kita tetap butuh Sastrawi untuk memproses QUERY pengguna
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)

# --- INIT NLP (Hanya untuk Query) ---
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()
factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()

# --- LOAD DATA YANG SUDAH JADI ---
try:
    with open("data_siap_pakai.json", encoding="utf-8") as f:
        doc_entries = json.load(f)
except FileNotFoundError:
    print("ERROR: File 'data_siap_pakai.json' tidak ditemukan.")
    print("Jalankan 'py bikin_data.py' terlebih dahulu!")
    exit()

# Ambil tokens yang sudah disimpan di JSON untuk membangun index BM25
corpus = [doc['tokens'] for doc in doc_entries]
bm25 = BM25(corpus)
print("Sistem Search Engine Siap! 🚀")

def preprocess_query(text):
    """Hanya memproses query yang diketik user"""
    if not text: return []
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text.split()

def highlight_text(original_text, query_stems):
    """
    Fungsi Kunci: Memberi tanda <mark> pada kata asli
    jika kata dasarnya (stem) cocok dengan query.
    """
    if not query_stems:
        return original_text
    
    words = original_text.split()
    highlighted_words = []
    
    for word in words:
        # Bersihkan kata dari tanda baca untuk dicek stem-nya
        clean_word = re.sub(r'[^a-zA-Z]', '', word).lower()
        # Cek stem kata tersebut
        word_stem = stemmer.stem(clean_word)
        
        # Jika stem-nya ada di list query user, kasih stabilo (mark)
        if word_stem in query_stems:
            highlighted_words.append(f"<mark style='background-color: #ffeeba; font-weight:bold;'>{word}</mark>")
        else:
            highlighted_words.append(word)
            
    return " ".join(highlighted_words)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/search')
def search():
    query_text = request.args.get("q", "").strip()
    if not query_text:
        return render_template("index.html")

    # 1. Preprocess Query
    query_tokens = preprocess_query(query_text)
    
    # Jika query tokens kosong, return kosong
    if not query_tokens:
        return render_template("results.html", query=query_text, results=[])

    # 2. Ranking BM25
    ranked_indices = bm25.ranked(query_tokens, 20)
    
    # 3. Grouping & Highlighting
    grouped_results = defaultdict(lambda: {
        "nama": "", "objek_pariwisata": [], "makanan_khas": []
    })

    # Konversi query tokens ke set untuk pengecekan cepat
    query_set = set(query_tokens)
    found_any_relevant = False

    for idx in ranked_indices:
        item = doc_entries[idx]
        
        # --- FILTER LOGIKA BARU ---
        # Cek apakah ada irisan (intersection) antara token query dan token dokumen
        doc_tokens = set(item['tokens'])
        
        # Jika tidak ada satu pun kata yang sama, skip dokumen ini
        if query_set.isdisjoint(doc_tokens):
            continue

        found_any_relevant = True
        
        # --- FITUR HIGHLIGHTING ---
        highlighted_content = highlight_text(item['original_konten'], query_tokens)
        
        display_item = {
            "nama": item['nama'],
            "konten": Markup(highlighted_content), 
            "gambar": item['gambar'],
            "type": item['type']
        }

        prov = item["provinsi"]
        grouped_results[prov]["nama"] = prov
        
        if item["type"] == "Wisata":
            # Cegah duplikat
            if not any(d['nama'] == display_item['nama'] for d in grouped_results[prov]["objek_pariwisata"]):
                grouped_results[prov]["objek_pariwisata"].append(display_item)
        elif item["type"] == "Kuliner":
            if not any(d['nama'] == display_item['nama'] for d in grouped_results[prov]["makanan_khas"]):
                grouped_results[prov]["makanan_khas"].append(display_item)

    # Jika setelah filter tidak ada hasil relevan, return list kosong
    if not found_any_relevant:
        results = []
    else:
        results = list(grouped_results.values())

    return render_template("results.html", query=query_text, results=results)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
# <Imports>
# coding=utf-8
import unicodedata

import nltk
import six
from nltk.corpus import mac_morpho

from bm25 import BM25


# <Function: normalize_terms>
def normalize_terms(terms):
    """Hapus diakritik dari term dan ubah ke huruf kecil"""
    # Di sini kalian dapat menambahkan fitur lain:
    # - menghapus stopwords
    # - menghapus angka
    # - stemming
    return [remove_diacritics(term).lower() for term in terms]


# <Function: remove_diacritics>
def remove_diacritics(text, encoding='utf8'):
    """Hapus diakritik dari bytestring atau unicode, mengembalikan string unicode"""
    nfkd_form = unicodedata.normalize('NFKD', to_unicode(text, encoding))
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode(encoding)


# <Function: to_unicode>
def to_unicode(text, encoding='utf8'):
    """Konversi string (bytestring dengan encoding tertentu atau unicode) ke unicode."""
    if isinstance(text, six.text_type):
        return text
    return text.decode(encoding)


# <Load Dataset>
# `news` adalah list yang berisi daftar token per kalimat
nltk.download('mac_morpho')  # Uncomment jika belum mengunduh resource
news = [normalize_terms(sentence) for sentence in mac_morpho.sents()]
print(repr(news[0]))


# <BM25 Indexing dan Query>
# Hanya menggunakan 1000 data pertama untuk contoh. Memproses semua 51397 data akan memakan waktu lama
bm25 = BM25(news[:1000])

# Buat query dan jalankan BM25
query = normalize_terms(nltk.word_tokenize('inflacao'))
for position, index in enumerate(bm25.ranked(query, 5)):
    print(f"{position} - {' '.join(news[index])}")
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords

# Download NLTK data (jalankan sekali)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Inisialisasi stemmer Bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Stopwords Bahasa Indonesia (opsional, bisa dikurangi)
id_stopwords = set(stopwords.words('indonesian'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Hapus karakter non-alfanumerik
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Tokenisasi
    tokens = text.split()
    # Stemming
    tokens = [stemmer.stem(token) for token in tokens]
    # Hapus stopwords (opsional – bisa dikomentari jika ingin akurasi lebih tinggi)
    # tokens = [token for token in tokens if token not in id_stopwords]
    return " ".join(tokens)
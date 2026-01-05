import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords

# SAFE download (tidak crash)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

try:
    id_stopwords = set(stopwords.words("indonesian"))
except:
    id_stopwords = set()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

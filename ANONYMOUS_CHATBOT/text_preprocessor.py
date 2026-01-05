"""
Text Preprocessor for Anonymous - Cybersecurity Chatbot
Full Bahasa Indonesia support with typo correction, cleaning, stemming, and stopword removal.
"""

import re
from typing import List, Union
import warnings
warnings.filterwarnings('ignore')

# Coba import Sastrawi
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    print("⚠️ Sastrawi tidak terinstal. Jalankan: pip install Sastrawi")

# Konfigurasi default (fallback jika rag_config.py tidak ada)
DEFAULT_CONFIG = {
    'typo_corrections': {
        # Typo umum cybersecurity & percakapan
        'phising': 'phishing',
        'phisingan': 'phishing',
        'malwer': 'malware',
        'ransomwer': 'ransomware',
        'firewol': 'firewall',
        'ddoss': 'ddos',
        'ddos attack': 'ddos',
        'ddosattack': 'ddos',
        'antivir': 'antivirus',
        'hacker': 'peretas',
        'hack': 'peretasan',
        'password': 'kata sandi',
        'akun': 'akun',
        'scam': 'penipuan',
        'scaming': 'penipuan',
        'virus': 'virus',
        'trojan': 'trojan',
        'botnet': 'botnet',
        'keylogger': 'keylogger',
        'exploit': 'eksploit',
        'vuln': 'kerentanan',
        'vulnerability': 'kerentanan',
        'zero day': 'zero-day',
        'zeroday': 'zero-day',
        'social engineering': 'rekayasa sosial',
        'pharming': 'pharming',
        'spyware': 'spyware',
        'adware': 'adware',
        'rootkit': 'rootkit',
        'backdoor': 'backdoor',
        # Typo percakapan
        'gimana': 'bagaimana',
        'kenapa': 'mengapa',
        'nggak': 'tidak',
        'gak': 'tidak',
        'ga': 'tidak',
        'makasih': 'terima kasih',
        'thanks': 'terima kasih'
    },
    'min_word_length': 2
}

class TextPreprocessor:
    """
    Unified text preprocessing for both training and inference.
    Handles edge cases like non-string inputs, empty strings, and invalid data.
    """
    
    def __init__(self):
        if not SASTRAWI_AVAILABLE:
            raise ImportError("Sastrawi diperlukan untuk preprocessing Bahasa Indonesia. "
                            "Silakan instal dengan: pip install Sastrawi")
        
        # Inisialisasi Sastrawi
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        
        # Muat konfigurasi
        try:
            from rag_config import PREPROCESS_CONFIG
            self.config = PREPROCESS_CONFIG
        except ImportError:
            print("⚠️ File rag_config.py tidak ditemukan. Menggunakan konfigurasi default.")
            self.config = DEFAULT_CONFIG
        
        self.typo_corrections = self.config['typo_corrections']
        self.min_word_length = self.config['min_word_length']
        
        # Stopwords tambahan (kata tanya umum yang tidak informatif untuk klasifikasi)
        self.custom_stopwords = {
            'apa', 'adalah', 'itu', 'ya', 'sih', 'dong', 'yah', 'kan', 'lah', 
            'kah', 'pun', 'tuh', 'gimana', 'bagaimana', 'kenapa', 'mengapa',
            'mengapa', 'bagaimana', 'berapa', 'siapa', 'kapan', 'di', 'ke',
            'dari', 'dan', 'atau', 'untuk', 'dengan', 'pada', 'ini', 'itu',
            'yang', 'ada', 'bisa', 'boleh', 'harus', 'akan', 'sudah', 'sedang'
        }

    def _ensure_string(self, text: Union[str, any]) -> str:
        """
        Pastikan input adalah string. Jika bukan, konversi atau kembalikan string kosong.
        Ini mencegah error seperti: 'dict' object has no attribute 'strip'
        """
        if isinstance(text, str):
            return text
        elif text is None:
            return ''
        else:
            # Jika bukan string (misal dict, list, int), konversi ke string
            # Tapi ini seharusnya tidak terjadi di pipeline yang benar
            print(f"⚠️ Peringatan: Input bukan string! Mengonversi {type(text)} ke string.")
            return str(text)

    def normalize_text(self, text: str) -> str:
        """Normalisasi typo dan istilah cybersecurity"""
        if not text:
            return ''
        
        text_lower = text.lower()
        words = text_lower.split()
        normalized_words = []
        
        for word in words:
            # Cek koreksi typo (termasuk frasa 2 kata)
            normalized_word = self.typo_corrections.get(word, word)
            normalized_words.append(normalized_word)
        
        return ' '.join(normalized_words)

    def clean_text(self, text: str) -> str:
        """Pembersihan teks dasar"""
        if not text:
            return ''
        
        # Lowercase
        text = text.lower()
        
        # Hapus URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Hapus email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Hanya izinkan huruf, angka, dan spasi
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Hapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def remove_short_words(self, text: str) -> str:
        """Hapus kata terlalu pendek"""
        if not text:
            return ''
        
        words = text.split()
        filtered = [w for w in words if len(w) >= self.min_word_length]
        return ' '.join(filtered)

    def preprocess(
        self,
        text: Union[str, any],
        remove_stopwords: bool = True,
        apply_stemming: bool = True
    ) -> str:
        """
        Pipeline preprocessing lengkap.
        
        Args:
            text: Input teks (string). Jika bukan string, akan dikonversi/diabaikan.
            remove_stopwords: Aktifkan penghapusan stopwords
            apply_stemming: Aktifkan stemming
            
        Returns:
            Teks yang sudah diproses (string)
        """
        # 🔒 Jamin input berupa string
        text = self._ensure_string(text)
        
        if not text or text.strip() == '':
            return ''
        
        # 1. Normalisasi typo
        text = self.normalize_text(text)
        
        # 2. Pembersihan
        text = self.clean_text(text)
        
        # 3. Hapus kata pendek
        text = self.remove_short_words(text)
        
        if not text:
            return ''
        
        # 4. Hapus stopwords kustom
        if remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in self.custom_stopwords]
            text = ' '.join(words)
            
            if text:
                # Gunakan stopword remover Sastrawi
                text = self.stopword_remover.remove(text)
        
        # 5. Stemming
        if apply_stemming and text:
            text = self.stemmer.stem(text)
        
        return text.strip()

    def preprocess_batch(
        self,
        texts: List[Union[str, any]],
        remove_stopwords: bool = True,
        apply_stemming: bool = True
    ) -> List[str]:
        """
        Preproses batch teks.
        Aman terhadap input non-string.
        """
        return [
            self.preprocess(text, remove_stopwords, apply_stemming)
            for text in texts
        ]


# ═══════════════════════════════════════════════════════════════════════════
# 🧪 TESTING (Opsional)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 80)
    print("🧹 TESTING TEXT PREPROCESSOR - ANONYMOUS CHATBOT")
    print("=" * 80)
    
    try:
        preprocessor = TextPreprocessor()
    except ImportError as e:
        print(f"❌ Error: {e}")
        exit(1)
    
    test_cases = [
        "Apa itu phising dan bagaimana cara mencegahnya?",
        "Gimana cara kerja malwer di komputer?",
        "Bisa jelasin tentang ransomwer dong",
        "Bedanya firewol sama antivirus tuh apa sih?",
        "Tolong bantu aku memahami ddoss attack",
        "Saya kena scam! Password akun saya dicuri!",
        "",  # kasus kosong
        None,  # kasus None
    ]
    
    print("\n📝 Hasil Preprocessing:")
    print("-" * 80)
    
    for i, text in enumerate(test_cases, 1):
        try:
            result = preprocessor.preprocess(text)
            print(f"\n{i}. Input: {repr(text)}")
            print(f"   Output: '{result}'")
        except Exception as e:
            print(f"\n{i}. Error processing {repr(text)}: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Testing selesai!")
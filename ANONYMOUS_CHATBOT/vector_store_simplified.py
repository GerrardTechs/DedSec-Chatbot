"""
═══════════════════════════════════════════════════════════════════════════════
🗄️ SIMPLIFIED VECTOR STORE - TF-IDF VERSION
Supports dataset format: {"intents": [{"intent": "...", "patterns": [...], "responses": [...]}, ...]}
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_preprocessor import TextPreprocessor
from rag_config import VECTOR_STORE_DIR

class VectorStore:
    """Vector store menggunakan TF-IDF untuk RAG sederhana"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.vectors = None
        self.texts = []          # teks asli (belum diproses)
        self.metadata = []       # metadata: intent, source, dll.
        
    def build_from_knowledge_base(self, knowledge_base_file):
        """Bangun vector store dari dataset augmented (format intents wrapper)"""
        print(f"\n{'='*80}")
        print("🗄️ BUILDING VECTOR STORE FROM DATASET")
        print('='*80)
        
        # Load dataset
        with open(knowledge_base_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        texts = []
        metadata = []
        
        # Deteksi format dataset
        if isinstance(data, dict) and 'intents' in data:
            intent_list = data['intents']
            print("✅ Format terdeteksi: {'intents': [...]}")
        elif isinstance(data, list):
            intent_list = data
            print("✅ Format terdeteksi: list langsung")
        else:
            raise ValueError("❌ Format tidak dikenali! Harus berupa list atau {'intents': [...]}")
        
        if not isinstance(intent_list, list):
            raise ValueError("❌ 'intents' harus berupa list!")
        
        # Ekstrak teks dari setiap intent
        for item in intent_list:
            if not isinstance(item, dict):
                continue
            
            intent = item.get('intent', 'unknown')
            
            # Ambil responses (prioritas utama)
            responses = item.get('responses', [])
            if isinstance(responses, str):
                responses = [responses]
            for resp in responses:
                if isinstance(resp, str) and resp.strip():
                    texts.append(resp.strip())
                    metadata.append({'intent': intent, 'source': 'response'})
            
            # Ambil patterns (opsional, untuk variasi query)
            patterns = item.get('patterns', [])
            if isinstance(patterns, str):
                patterns = [patterns]
            for pat in patterns:
                if isinstance(pat, str) and pat.strip():
                    texts.append(pat.strip())
                    metadata.append({'intent': intent, 'source': 'pattern'})
        
        print(f"✅ Berhasil mengekstrak {len(texts)} teks dari {len(intent_list)} intent")
        
        if len(texts) == 0:
            raise ValueError("❌ Tidak ada teks valid ditemukan di dataset!")
        
        # Preprocessing
        print("🧹 Preprocessing teks...")
        preprocessed_texts = self.preprocessor.preprocess_batch(texts)
        
        # Filter teks kosong setelah preprocessing
        filtered = [(t, m) for t, m in zip(preprocessed_texts, metadata) if t.strip()]
        if not filtered:
            raise ValueError("❌ Semua teks menjadi kosong setelah preprocessing!")
        
        final_preprocessed, final_metadata = zip(*filtered)
        final_preprocessed = list(final_preprocessed)
        final_metadata = list(final_metadata)
        final_original_texts = [t for t, m in zip(texts, metadata) if t.strip()][:len(final_preprocessed)]
        
        # Bangun TF-IDF vectors
        print("🔢 Membangun vektor TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
        self.vectors = self.vectorizer.fit_transform(final_preprocessed)
        self.texts = final_original_texts
        self.metadata = final_metadata
        
        print(f"✅ Vector store siap! {len(self.texts)} dokumen, vocabulary: {len(self.vectorizer.vocabulary_)}")
    
    def search(self, query: str, top_k: int = 3):
        """Cari teks paling relevan berdasarkan query"""
        if not query or not isinstance(query, str):
            return []
        
        # Preprocess query
        preprocessed_query = self.preprocessor.preprocess(query)
        if not preprocessed_query:
            return []
        
        # Vectorize dan hitung similarity
        query_vector = self.vectorizer.transform([preprocessed_query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Ambil top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format hasil
        results = []
        for idx in top_indices:
            results.append({
                'text': self.texts[idx],
                'metadata': self.metadata[idx],
                'score': float(similarities[idx])
            })
        return results
    
    def save(self, save_dir=None):
        """Simpan vector store ke disk"""
        if save_dir is None:
            save_dir = VECTOR_STORE_DIR
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.vectorizer, save_dir / 'vectorizer.pkl')
        joblib.dump(self.vectors, save_dir / 'vectors.pkl')
        joblib.dump(self.texts, save_dir / 'texts.pkl')
        joblib.dump(self.metadata, save_dir / 'metadata.pkl')
        
        print(f"✅ Vector store disimpan di: {save_dir}")
    
    def load(self, load_dir=None):
        """Muat vector store dari disk"""
        if load_dir is None:
            load_dir = VECTOR_STORE_DIR
        
        self.vectorizer = joblib.load(load_dir / 'vectorizer.pkl')
        self.vectors = joblib.load(load_dir / 'vectors.pkl')
        self.texts = joblib.load(load_dir / 'texts.pkl')
        self.metadata = joblib.load(load_dir / 'metadata.pkl')
        
        print(f"✅ Vector store dimuat dari: {load_dir}")

# ═══════════════════════════════════════════════════════════════════════════
# 🧪 TESTING
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🗄️ VECTOR STORE - TESTING")
    print("="*80)
    
    # Inisialisasi
    vs = VectorStore()
    
    # Bangun dari dataset
    vs.build_from_knowledge_base('dataset_augmented.json')
    
    # Simpan
    vs.save()
    
    # Uji pencarian
    print(f"\n{'='*80}")
    print("🔍 TESTING PENCARIAN")
    print('='*80)
    
    test_queries = [
        "apa itu phishing",
        "jelaskan cybersecurity",
        "contoh serangan siber"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        results = vs.search(query, top_k=2)
        
        if not results:
            print("   ❌ Tidak ada hasil")
            continue
            
        for i, r in enumerate(results, 1):
            # Pastikan text bisa di-display sebagai string
            text_display = str(r['text']).replace('\n', ' ')[:120]
            intent_name = r['metadata'].get('intent', 'N/A')
            print(f"\n   {i}. Skor: {r['score']:.4f}")
            print(f"      Intent: {intent_name}")
            print(f"      Teks: {text_display}...")
    
    print(f"\n{'='*80}")
    print("✅ SEMUA TES SELESAI!")
    print('='*80)
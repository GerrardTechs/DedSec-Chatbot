"""
═══════════════════════════════════════════════════════════════════════════════
🗄️ VECTOR STORE - FAISS IMPLEMENTATION
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import numpy as np
import faiss
import joblib
from pathlib import Path
from typing import List, Dict

from embedding_service import EmbeddingService
from rag_config import VECTOR_STORE_CONFIG, VECTOR_STORE_DIR

class VectorStore:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.index = None
        self.texts = []
        self.metadata = []
        
    def build_from_knowledge_base(self, knowledge_base_file):
        """Build FAISS index from knowledge base"""
        print(f"\n{'='*80}")
        print("🗄️ BUILDING VECTOR STORE")
        print('='*80)
        
        # Load knowledge base
        with open(knowledge_base_file, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        
        # Extract texts and metadata
        texts = []
        metadata = []
        
        for intent, data in kb.items():
            if isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, list):
                        for val in values:
                            texts.append(val)
                            metadata.append({
                                'intent': intent,
                                'type': key
                            })
        
        print(f"✅ Extracted {len(texts)} texts from knowledge base")
        
        # Generate embeddings
        print("🔤 Generating embeddings...")
        embeddings = self.embedding_service.encode(texts, show_progress=True)
        
        # Build FAISS index
        dimension = VECTOR_STORE_CONFIG['dimension']
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.texts = texts
        self.metadata = metadata
        
        print(f"✅ FAISS index built with {len(texts)} vectors")
        
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for similar texts"""
        if top_k is None:
            top_k = VECTOR_STORE_CONFIG['top_k']
        
        # Encode query
        query_vector = self.embedding_service.encode(query, show_progress=False)
        
        # Search
        distances, indices = self.index.search(
            query_vector.reshape(1, -1).astype('float32'),
            top_k
        )
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                'text': self.texts[idx],
                'metadata': self.metadata[idx],
                'score': float(distance)
            })
        
        return results
    
    def save(self, save_dir=None):
        """Save vector store"""
        if save_dir is None:
            save_dir = VECTOR_STORE_DIR
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_dir / 'faiss.index'))
        
        # Save texts and metadata
        joblib.dump(self.texts, save_dir / 'texts.pkl')
        joblib.dump(self.metadata, save_dir / 'metadata.pkl')
        
        print(f"✅ Vector store saved to: {save_dir}")
    
    def load(self, load_dir=None):
        """Load vector store"""
        if load_dir is None:
            load_dir = VECTOR_STORE_DIR
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_dir / 'faiss.index'))
        
        # Load texts and metadata
        self.texts = joblib.load(load_dir / 'texts.pkl')
        self.metadata = joblib.load(load_dir / 'metadata.pkl')
        
        print(f"✅ Vector store loaded from: {load_dir}")

if __name__ == '__main__':
    vs = VectorStore()
    vs.build_from_knowledge_base('dataset_augmented.json')
    vs.save()
    
    # Test search
    results = vs.search("apa itu phishing")
    for r in results:
        print(f"Score: {r['score']:.2f} | {r['text'][:100]}...")
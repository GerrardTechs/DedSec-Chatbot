"""
═══════════════════════════════════════════════════════════════════════════════
🔤 EMBEDDING SERVICE
═══════════════════════════════════════════════════════════════════════════════
Generate embeddings using Sentence Transformers for FAISS vector store
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import warnings
warnings.filterwarnings('ignore')

from rag_config import EMBEDDING_CONFIG

class EmbeddingService:
    """
    Generate embeddings for texts using Sentence Transformers
    """
    
    def __init__(self):
        print(f"\n{'='*80}")
        print("🔤 INITIALIZING EMBEDDING SERVICE")
        print('='*80)
        
        model_name = EMBEDDING_CONFIG['model_name']
        device = EMBEDDING_CONFIG['device']
        
        print(f"📦 Loading model: {model_name}")
        print(f"🖥️  Device: {device}")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Model loaded successfully!")
        print(f"✅ Embedding dimension: {self.dimension}")
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = None,
        show_progress: bool = True,
        normalize: bool = None
    ) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if batch_size is None:
            batch_size = EMBEDDING_CONFIG['batch_size']
        
        if normalize is None:
            normalize = EMBEDDING_CONFIG['normalize_embeddings']
        
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
            was_single = True
        else:
            was_single = False
        
        # Encode
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        # Return single embedding if input was single
        if was_single:
            return embeddings[0]
        
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.encode(text1, show_progress=False)
        emb2 = self.encode(text2, show_progress=False)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

# ═══════════════════════════════════════════════════════════════════════════
# 🧪 TESTING
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🔤 EMBEDDING SERVICE - TESTING")
    print("="*80)
    
    # Initialize
    service = EmbeddingService()
    
    # Test single encoding
    print(f"\n{'='*80}")
    print("🧪 Test 1: Single Text Encoding")
    print('='*80)
    
    text = "Apa itu phishing?"
    embedding = service.encode(text, show_progress=False)
    
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 5): {embedding[:5]}")
    
    # Test batch encoding
    print(f"\n{'='*80}")
    print("🧪 Test 2: Batch Encoding")
    print('='*80)
    
    texts = [
        "Apa itu phishing?",
        "Bagaimana cara mencegah malware?",
        "Jelaskan tentang ransomware"
    ]
    
    embeddings = service.encode(texts, show_progress=True)
    
    print(f"Number of texts: {len(texts)}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test similarity
    print(f"\n{'='*80}")
    print("🧪 Test 3: Similarity Calculation")
    print('='*80)
    
    text1 = "Apa itu phishing?"
    text2 = "Jelaskan tentang phishing attack"
    text3 = "Bagaimana cara mencegah ransomware?"
    
    sim_12 = service.similarity(text1, text2)
    sim_13 = service.similarity(text1, text3)
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity 1-2: {sim_12:.4f}")
    print()
    print(f"Text 1: {text1}")
    print(f"Text 3: {text3}")
    print(f"Similarity 1-3: {sim_13:.4f}")
    
    print(f"\n{'='*80}")
    print("✅ ALL TESTS COMPLETED!")
    print('='*80)

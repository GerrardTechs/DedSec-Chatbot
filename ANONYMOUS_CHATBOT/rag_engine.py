"""
═══════════════════════════════════════════════════════════════════════════════
🚀 RAG ENGINE - COMPLETE PIPELINE
═══════════════════════════════════════════════════════════════════════════════
"""

from intent_classifier import IntentClassifier
from vector_store import VectorStore
from llm_service import LLMService
from rag_config import RAG_CONFIG

class RAGEngine:
    def __init__(self):
        print(f"\n{'='*80}")
        print("🚀 INITIALIZING RAG ENGINE")
        print('='*80)
        
        # Load components
        self.intent_classifier = IntentClassifier()
        self.intent_classifier.load()
        
        self.vector_store = VectorStore()
        self.vector_store.load()
        
        self.llm = LLMService()
        
        print(f"✅ RAG Engine initialized!")
    
    def chat(self, query: str) -> Dict:
        """Complete RAG pipeline"""
        
        # Step 1: Intent classification
        intent_result = self.intent_classifier.predict(query)
        
        # Step 2: Vector search
        contexts = self.vector_store.search(query, top_k=3)
        
        # Step 3: LLM generation
        response = self.llm.generate(query, contexts)
        
        return {
            'query': query,
            'intent': intent_result['intent'],
            'confidence': intent_result['confidence'],
            'contexts': contexts,
            'response': response
        }

if __name__ == '__main__':
    engine = RAGEngine()
    
    result = engine.chat("apa itu phishing")
    print(f"\nQuery: {result['query']}")
    print(f"Intent: {result['intent']} ({result['confidence']:.2%})")
    print(f"Response: {result['response']}")
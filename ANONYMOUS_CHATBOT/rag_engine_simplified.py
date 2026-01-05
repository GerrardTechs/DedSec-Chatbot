"""
══════════════════════════════════════════════════════════════════════════════
🛡️ SIMPLIFIED RAG ENGINE FOR ANONYMOUS CHATBOT
Compatible with enhanced IntentClassifier (loads intent_responses)
══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import random
from pathlib import Path

from intent_classifier import IntentClassifier
from vector_store_simplified import VectorStore
from rag_config import INTENT_MODEL_DIR, VECTOR_STORE_DIR

class RAGEngine:
    def __init__(self):
        print("\n" + "="*80)
        print("🚀 INITIALIZING SIMPLIFIED RAG ENGINE")
        print("="*80)
        
        # 1. Load Intent Classifier
        print("📦 Loading Intent Classifier...")
        self.intent_classifier = IntentClassifier()
        self.intent_classifier.load()
        print("✅ Intent Classifier loaded")
        
        # 2. Load Vector Store
        print("📦 Loading Vector Store...")
        self.vector_store = VectorStore()
        self.vector_store.load()
        print("✅ Vector Store loaded")
        
        # 3. (Opsional) Load Response Templates from file
        self.response_templates = {}
        templates_file = Path("response_templates.json")
        if templates_file.exists():
            try:
                with open(templates_file, 'r', encoding='utf-8') as f:
                    raw_templates = json.load(f)
                for intent, responses in raw_templates.items():
                    if isinstance(responses, str):
                        self.response_templates[intent] = [responses]
                    elif isinstance(responses, list):
                        self.response_templates[intent] = [str(r) for r in responses if isinstance(r, (str, int, float))]
                    elif isinstance(responses, dict):
                        candidate = responses.get('response') or responses.get('text') or responses.get('answer', '')
                        self.response_templates[intent] = [str(candidate)] if candidate else []
                    else:
                        self.response_templates[intent] = [str(responses)]
                print("✅ Response Templates loaded and normalized")
            except Exception as e:
                print(f"⚠️ Gagal memuat response_templates.json: {e}")
        else:
            print("ℹ️ Tidak ada response_templates.json — gunakan respons dari intent classifier")

    def _get_base_response(self, intent: str) -> str:
        """Dapatkan respons dasar sebagai STRING dari intent classifier atau fallback"""
        # 1. Coba ambil dari intent_responses di classifier
        if hasattr(self.intent_classifier, 'intent_responses'):
            responses = self.intent_classifier.intent_responses.get(intent, [])
            if isinstance(responses, list) and responses:
                valid_responses = [str(r).strip() for r in responses if str(r).strip()]
                if valid_responses:
                    return random.choice(valid_responses)
        
        # 2. Fallback internal
        fallback_map = {
            "greeting": "Halo! Saya Anonymous, chatbot keamanan siber. Ada yang bisa saya bantu?",
            "goodbye": "Sampai jumpa! Tetap waspada di dunia digital.",
            "thanks": "Sama-sama! Jaga selalu keamanan digitalmu.",
            "fallback": "Maaf, saya belum memahami pertanyaan tersebut."
        }
        
        return fallback_map.get(intent, f"Saya memiliki informasi tentang '{intent}', tapi belum siap menjawab.")

    def _format_context_info(self, contexts: list) -> str:
        """Format konteks tambahan dari vector store"""
        if not contexts:
            return ""
        useful = [ctx for ctx in contexts if ctx.get('score', 0) > 0.1][:2]
        if not useful:
            return ""
        info_parts = []
        for ctx in useful:
            text = str(ctx.get('text', '')).strip()
            if text and len(text) > 20:
                info_parts.append(text)
        if not info_parts:
            return ""
        return "\n\nℹ️ Informasi tambahan:\n• " + "\n• ".join(info_parts)

    def generate_response(self, intent: str, contexts: list, confidence: float) -> str:
        """Bangun respons akhir (pastikan semua string)"""
        base_response = self._get_base_response(intent)
        additional_info = self._format_context_info(contexts)
        if not isinstance(base_response, str):
            base_response = str(base_response)
        if not isinstance(additional_info, str):
            additional_info = str(additional_info)
        response = base_response + additional_info
        if confidence < 0.4:
            response += "\n\n🔍 Catatan: Saya kurang yakin dengan maksud pertanyaan ini."
        return response

    def chat(self, query: str) -> dict:
        """Fungsi utama untuk memproses pertanyaan pengguna"""
        try:
            pred_result = self.intent_classifier.predict(query)
            intent = str(pred_result['intent'])
            confidence = float(pred_result['confidence'])
            contexts = self.vector_store.search(query, top_k=3)
            response = self.generate_response(intent, contexts, confidence)
            return {
                "query": query,
                "intent": intent,
                "confidence": round(confidence, 4),
                "response": response,
                "contexts_used": len([c for c in contexts if c.get('score', 0) > 0.1])
            }
        except Exception as e:
            print(f"❌ ERROR during chat: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query": query,
                "intent": "error",
                "confidence": 0.0,
                "response": "Maaf, terjadi kesalahan saat memproses permintaan Anda.",
                "contexts_used": 0
            }

# ═══════════════════════════════════════════════════════════════════════════
# 🧪 MAIN TESTING
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 SIMPLIFIED RAG ENGINE - TESTING")
    print("="*80)
    
    engine = RAGEngine()
    
    print("\n" + "="*80)
    print("🧪 TESTING QUERIES")
    print("="*80)
    
    test_queries = [
        "apa itu phishing",
        "bagaimana cara mencegah malware",
        "apa itu ransomware",
        "tips membuat password yang aman",
        "hai",
        "terima kasih"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        try:
            result = engine.chat(query)
            print(f"🎯 Intent: {result['intent']} (confidence: {result['confidence']:.2%})")
            print(f"💬 Response:\n{result['response']}")
            if result['contexts_used'] > 0:
                print(f"📚 Konteks tambahan digunakan: {result['contexts_used']}")
        except Exception as inner_e:
            print(f"❌ Gagal memproses query: {inner_e}")
    
    print("\n" + "="*80)
    print("✅ TESTING SELESAI!")
    print("="*80)
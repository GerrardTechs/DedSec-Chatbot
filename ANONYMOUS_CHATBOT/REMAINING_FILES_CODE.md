# 🚀 REMAINING RAG FILES - COMPLETE CODE

All remaining files with production-ready code. Copy each section into separate files.

---

## FILE 5: vector_store.py

```python
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
```

---

## FILE 6: llm_service.py

```python
"""
═══════════════════════════════════════════════════════════════════════════════
🤖 LLM SERVICE - QWEN 2.5 INTEGRATION
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

from rag_config import LLM_CONFIG, PROMPT_TEMPLATES

class LLMService:
    def __init__(self):
        print(f"\n{'='*80}")
        print("🤖 INITIALIZING LLM SERVICE")
        print('='*80)
        
        model_name = LLM_CONFIG['model_name']
        device = LLM_CONFIG['device']
        
        print(f"📦 Loading model: {model_name}")
        print(f"🖥️  Device: {device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with quantization if specified
        if LLM_CONFIG['quantization'] == '4bit':
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map='auto'
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                device_map='auto'
            )
        
        print(f"✅ LLM loaded successfully!")
    
    def generate(self, query: str, contexts: List[Dict]) -> str:
        """Generate response with RAG contexts"""
        
        # Build context text
        context_text = "\n\n".join([ctx['text'] for ctx in contexts])
        
        # Build prompt
        system_prompt = PROMPT_TEMPLATES['system_prompt']
        user_prompt = PROMPT_TEMPLATES['user_prompt_template'].format(
            query=query,
            context=context_text
        )
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors='pt').to(self.model.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=LLM_CONFIG['max_new_tokens'],
            temperature=LLM_CONFIG['temperature'],
            top_p=LLM_CONFIG['top_p'],
            repetition_penalty=LLM_CONFIG['repetition_penalty'],
            do_sample=LLM_CONFIG['do_sample']
        )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        if "Jawaban:" in response:
            response = response.split("Jawaban:")[-1].strip()
        elif full_prompt in response:
            response = response.replace(full_prompt, "").strip()
        
        return response

if __name__ == '__main__':
    llm = LLMService()
    
    contexts = [{'text': 'Phishing adalah teknik penipuan siber.'}]
    response = llm.generate("apa itu phishing", contexts)
    print(f"Response: {response}")
```

---

## FILE 7: rag_engine.py

```python
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
```

---

## FILE 8: evaluation.py

```python
"""
═══════════════════════════════════════════════════════════════════════════════
📊 EVALUATION - COMPREHENSIVE METRICS
═══════════════════════════════════════════════════════════════════════════════
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json

from intent_classifier import IntentClassifier
from rag_config import EVAL_CONFIG, CONFUSION_MATRIX_PATH, CLASSIFICATION_REPORT_PATH

class Evaluator:
    def __init__(self):
        self.classifier = IntentClassifier()
    
    def plot_confusion_matrix(self, cm, labels, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=EVAL_CONFIG['confusion_matrix_figsize'])
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix - Intent Classification')
        plt.ylabel('True Intent')
        plt.xlabel('Predicted Intent')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved: {save_path}")
        
        plt.close()
    
    def save_classification_report(self, report, save_path=None):
        """Save classification report"""
        if save_path is None:
            save_path = CLASSIFICATION_REPORT_PATH
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            for intent, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    f.write(f"{intent}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {metrics['support']}\n\n")
        
        print(f"✅ Classification report saved: {save_path}")
    
    def evaluate_model(self, X_test, y_test, save_plots=True):
        """Complete evaluation"""
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE EVALUATION")
        print('='*80)
        
        # Predictions
        y_pred = self.classifier.model.predict(X_test)
        
        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=self.classifier.intent_labels)
        
        # Check thresholds
        accuracy = report['accuracy']
        f1_macro = report['macro avg']['f1-score']
        
        print(f"\n✅ Accuracy: {accuracy:.4f} (Threshold: {EVAL_CONFIG['min_accuracy']})")
        print(f"✅ F1-Score (Macro): {f1_macro:.4f} (Threshold: {EVAL_CONFIG['min_f1_score']})")
        
        if accuracy >= EVAL_CONFIG['min_accuracy']:
            print("✅ Accuracy threshold MET!")
        else:
            print("❌ Accuracy threshold NOT met")
        
        if f1_macro >= EVAL_CONFIG['min_f1_score']:
            print("✅ F1-Score threshold MET!")
        else:
            print("❌ F1-Score threshold NOT met")
        
        # Save
        if save_plots:
            self.plot_confusion_matrix(cm, self.classifier.intent_labels, CONFUSION_MATRIX_PATH)
            self.save_classification_report(report)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }

if __name__ == '__main__':
    evaluator = Evaluator()
    # Load test data and evaluate
    print("Run training_pipeline.py first to generate test data")
```

---

CONTINUE IN NEXT MESSAGE...

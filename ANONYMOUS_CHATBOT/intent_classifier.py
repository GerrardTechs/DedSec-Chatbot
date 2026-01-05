"""
═══════════════════════════════════════════════════════════════════════════════
🎯 INTENT CLASSIFIER - WITH 80/20 SPLIT + RESPONSE STORAGE
═══════════════════════════════════════════════════════════════════════════════
Intent classification with ensemble models and full response saving/loading
"""

import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from text_preprocessor import TextPreprocessor
from rag_config import INTENT_CONFIG, INTENT_MODEL_DIR

class IntentClassifier:
    """
    Intent classification with ensemble models
    Implements 80/20 train/test split and saves responses for inference
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.intent_labels = None
        self.intent_responses = {}  # ← Menyimpan respons teks per intent
        
        # Config
        self.test_size = INTENT_CONFIG['test_size']
        self.random_state = INTENT_CONFIG['random_state']
        self.use_ensemble = INTENT_CONFIG['use_ensemble']
        self.training_data_file = 'dataset_augmented.json'
        
    def load_data(self, data_file):
        """Load training data from JSON"""
        print(f"\n{'='*80}")
        print("📚 LOADING TRAINING DATA")
        print('='*80)
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to lists
        texts = []
        labels = []
        
        # Handle format: {"intents": [...]} or list directly
        if isinstance(data, dict) and 'intents' in data:
            intent_list = data['intents']
        elif isinstance(data, list):
            intent_list = data
        else:
            raise ValueError("Format dataset tidak didukung!")
        
        for item in intent_list:
            if isinstance(item, dict) and 'intent' in item and 'patterns' in item:
                intent = item['intent']
                for pattern in item['patterns']:
                    if isinstance(pattern, str):
                        texts.append(pattern)
                        labels.append(intent)
        
        print(f"✅ Loaded {len(texts)} samples across {len(set(labels))} intents")
        return texts, labels, intent_list  # ← Kembalikan juga intent_list untuk simpan responses
    
    def preprocess_data(self, texts):
        """Preprocess all texts"""
        print(f"\n{'='*80}")
        print("🧹 PREPROCESSING DATA")
        print('='*80)
        preprocessed = self.preprocessor.preprocess_batch(texts)
        print(f"✅ Preprocessed {len(preprocessed)} texts")
        return preprocessed
    
    def split_data(self, texts, labels):
        """Split data into train and test sets (80/20)"""
        print(f"\n{'='*80}")
        print("📊 SPLITTING DATA (80/20)")
        print('='*80)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=None
        )
        print(f"✅ Training set: {len(X_train)} samples ({(1-self.test_size)*100:.0f}%)")
        print(f"✅ Testing set: {len(X_test)} samples ({self.test_size*100:.0f}%)")
        return X_train, X_test, y_train, y_test
    
    def vectorize_data(self, X_train, X_test):
        """Convert texts to TF-IDF vectors"""
        print(f"\n{'='*80}")
        print("🔢 VECTORIZING DATA")
        print('='*80)
        tfidf_params = INTENT_CONFIG['tfidf_params']
        self.vectorizer = TfidfVectorizer(**tfidf_params)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        print(f"✅ Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"✅ Training vectors: {X_train_vec.shape}")
        print(f"✅ Testing vectors: {X_test_vec.shape}")
        return X_train_vec, X_test_vec
    
    def build_model(self):
        """Build ensemble or single model"""
        print(f"\n{'='*80}")
        print("🤖 BUILDING MODEL")
        print('='*80)
        if self.use_ensemble:
            print("📦 Building Ensemble Model...")
            lr = LogisticRegression(max_iter=1000, random_state=self.random_state, class_weight='balanced')
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced', n_jobs=-1)
            svc = SVC(kernel='linear', probability=True, random_state=self.random_state, class_weight='balanced')
            self.model = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svc', svc)], voting='soft')
            print("✅ Ensemble model created (LR + RF + SVC)")
        else:
            print("📦 Building Logistic Regression Model...")
            self.model = LogisticRegression(max_iter=1000, random_state=self.random_state, class_weight='balanced')
            print("✅ Logistic Regression model created")
        return self.model
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\n{'='*80}")
        print("🎓 TRAINING MODEL")
        print('='*80)
        self.model.fit(X_train, y_train)
        self.intent_labels = sorted(list(set(y_train)))
        print(f"✅ Model trained successfully!")
        print(f"✅ Intent labels: {len(self.intent_labels)}")
        
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        print(f"\n{'='*80}")
        print("📊 EVALUATING MODEL")
        print('='*80)
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        report = classification_report(
    y_test,
    y_pred,
    labels=self.intent_labels,  # ← tambahkan ini
    target_names=self.intent_labels,
    output_dict=True,
    zero_division=0
)
        print(f"\n📋 Classification Report:")
        print(f"{'Intent':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print('-' * 65)
        for intent in self.intent_labels:
            if intent in report:
                metrics = report[intent]
                print(f"{intent:<25} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f}")
        print('-' * 65)
        print(f"{'Macro Avg':<25} {report['macro avg']['precision']:<12.4f} {report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f}")
        print(f"{'Weighted Avg':<25} {report['weighted avg']['precision']:<12.4f} {report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f}")
        cm = confusion_matrix(y_test, y_pred, labels=self.intent_labels)
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    def predict(self, text):
        """Predict intent for a single text"""
        preprocessed = self.preprocessor.preprocess(text)
        vector = self.vectorizer.transform([preprocessed])
        intent = self.model.predict(vector)[0]
        proba = self.model.predict_proba(vector)[0]
        confidence = float(max(proba))
        intent_probas = {label: float(prob) for label, prob in zip(self.intent_labels, proba)}
        return {
            'intent': intent,
            'confidence': confidence,
            'all_probabilities': intent_probas
        }
    
    def save(self, model_dir=None):
        """Save model and artifacts (including responses)"""
        if model_dir is None:
            model_dir = INTENT_MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*80}")
        print("💾 SAVING MODEL")
        print('='*80)
        # Save model
        joblib.dump(self.model, model_dir / 'intent_model.pkl')
        print(f"✅ Saved: intent_model.pkl")
        # Save vectorizer
        joblib.dump(self.vectorizer, model_dir / 'vectorizer.pkl')
        print(f"✅ Saved: vectorizer.pkl")
        # Save preprocessor
        joblib.dump(self.preprocessor, model_dir / 'preprocessor.pkl')
        print(f"✅ Saved: preprocessor.pkl")
        # Save intent labels
        with open(model_dir / 'intent_labels.json', 'w', encoding='utf-8') as f:
            json.dump(self.intent_labels, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: intent_labels.json")
        # Save intent responses
        try:
            with open(self.training_data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            if isinstance(raw_data, dict) and 'intents' in raw_data:
                intents_list = raw_data['intents']
            else:
                intents_list = raw_data if isinstance(raw_data, list) else []
            intent_resp = {}
            for item in intents_list:
                if isinstance(item, dict) and 'intent' in item and 'responses' in item:
                    intent_resp[item['intent']] = item['responses']
            joblib.dump(intent_resp, model_dir / 'intent_responses.pkl')
            print(f"✅ Saved: intent_responses.pkl")
        except Exception as e:
            print(f"⚠️ Gagal simpan intent_responses: {e}")
        print(f"\n✅ All artifacts saved to: {model_dir}")
    
    def load(self, model_dir=None):
        """Load model and artifacts (including responses)"""
        if model_dir is None:
            model_dir = INTENT_MODEL_DIR
        print(f"\n{'='*80}")
        print("📂 LOADING MODEL")
        print('='*80)
        # Load model
        self.model = joblib.load(model_dir / 'intent_model.pkl')
        print(f"✅ Loaded: intent_model.pkl")
        # Load vectorizer
        self.vectorizer = joblib.load(model_dir / 'vectorizer.pkl')
        print(f"✅ Loaded: vectorizer.pkl")
        # Load preprocessor
        self.preprocessor = joblib.load(model_dir / 'preprocessor.pkl')
        print(f"✅ Loaded: preprocessor.pkl")
        # Load intent labels
        with open(model_dir / 'intent_labels.json', 'r', encoding='utf-8') as f:
            self.intent_labels = json.load(f)
        print(f"✅ Loaded: intent_labels.json")
        # Load intent responses
        resp_path = model_dir / 'intent_responses.pkl'
        if resp_path.exists():
            self.intent_responses = joblib.load(resp_path)
            print(f"✅ Loaded: intent_responses.pkl")
        else:
            self.intent_responses = {}
            print("⚠️ intent_responses.pkl not found (fallback will be used)")
        print(f"\n✅ Model loaded from: {model_dir}")

# ═══════════════════════════════════════════════════════════════════════════
# 🧪 TESTING
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🎯 INTENT CLASSIFIER - TRAINING & EVALUATION")
    print("="*80)
    
    classifier = IntentClassifier()
    
    import os
    data_file = classifier.training_data_file
    if not os.path.exists(data_file):
        print(f"\n❌ ERROR: File '{data_file}' not found!")
        exit(1)
    
    print(f"\n✅ Using training data: {data_file}")
    
    # Load data (now returns intent_list too)
    texts, labels, intent_list = classifier.load_data(data_file)
    
    if len(texts) == 0:
        print(f"\n❌ ERROR: No samples loaded!")
        exit(1)
    
    # Preprocess
    preprocessed_texts = classifier.preprocess_data(texts)
    
    # Split
    X_train, X_test, y_train, y_test = classifier.split_data(preprocessed_texts, labels)
    
    # Vectorize
    X_train_vec, X_test_vec = classifier.vectorize_data(X_train, X_test)
    
    # Build & Train
    classifier.build_model()
    classifier.train(X_train_vec, y_train)
    
    # Evaluate
    results = classifier.evaluate(X_test_vec, y_test)
    
    # Save
    classifier.save()
    
    # Test predictions
    print(f"\n{'='*80}")
    print("🧪 TESTING PREDICTIONS")
    print('='*80)
    test_queries = ["apa itu phishing", "cara mencegah ransomware"]
    for query in test_queries:
        result = classifier.predict(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.2%}")
    
    print(f"\n{'='*80}")
    print("✅ ALL TESTS COMPLETED!")
    print('='*80)
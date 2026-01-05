"""
TRAINING SCRIPT - SKLEARN VERSION (FINAL FIXED)
100% Compatible dengan format Anda
"""

import json
import numpy as np
import pickle
import random
import nltk
import os
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

print("""
╔═══════════════════════════════════════════════════════════╗
║  ANONYMOUS CHATBOT - TRAINING (SKLEARN VERSION)          ║
║  NO TENSORFLOW REQUIRED - Menggunakan Random Forest      ║
╚═══════════════════════════════════════════════════════════╝
""")

# Initialize
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', ';', ':']

# Load intents
print("📂 Loading intents file...")

intent_files = [
    'dataset_augmented.json',
    'intent_training_data_expanded.json',
    'data_augmented.json',
    'intents.json',
]

intents = None
loaded_file = None

for filename in intent_files:
    if not os.path.exists(filename):
        continue
        
    try:
        print(f"  → Trying {filename}...")
        with open(filename, encoding='utf-8') as file:
            data = json.load(file)
            
        if 'intents' in data:
            intents = data
            loaded_file = filename
            print(f"✓ Successfully loaded: {filename}")
            print(f"✓ Total intents: {len(intents['intents'])}")
            total_patterns = sum(len(intent.get('patterns', [])) for intent in intents['intents'])
            print(f"✓ Total patterns: {total_patterns}")
            break
            
    except Exception as e:
        print(f"  ✗ Error with {filename}: {e}")
        continue

if intents is None:
    print("\n❌ ERROR: No valid intents file found!")
    print("\nSearched for:")
    for f in intent_files:
        print(f"  - {f}")
    exit(1)

# Preprocessing
print("\n📊 Preprocessing data...")
for intent in intents['intents']:
    # Support both 'intent' and 'tag' keys
    intent_tag = intent.get('tag') or intent.get('intent')
    if not intent_tag:
        print(f"⚠️  Skipping intent without tag/intent key: {intent}")
        continue
    
    patterns = intent.get('patterns', [])
    if not patterns:
        print(f"⚠️  Skipping {intent_tag}: no patterns")
        continue
    
    for pattern in patterns:
        # Tokenize
        word_list = nltk.word_tokenize(pattern.lower())
        words.extend(word_list)
        # Add to documents
        documents.append((word_list, intent_tag))
        # Add to classes
        if intent_tag not in classes:
            classes.append(intent_tag)

# Lemmatize and clean
print("🔧 Lemmatizing words...")
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

print(f"\n✓ Preprocessing completed:")
print(f"  - {len(documents)} documents")
print(f"  - {len(classes)} classes")
print(f"  - {len(words)} unique words")

if len(documents) < 10:
    print("\n⚠️  WARNING: Very few training samples!")
    print("   Consider adding more patterns to improve accuracy")

# Create training data
print("\n🔧 Creating training data...")
training_x = []
training_y = []

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create bag of words
    for word in words:
        bag.append(1 if word in word_patterns else 0)
    
    training_x.append(bag)
    training_y.append(document[1])

# Convert to numpy arrays
training_x = np.array(training_x)
training_y = np.array(training_y)

print(f"✓ Training data shape: X{training_x.shape}, Y{training_y.shape}")

# Split data - handle small datasets
test_size = 0.2 if len(documents) > 20 else 0.1

try:
    # Try with stratify first
    X_train, X_test, y_train, y_test = train_test_split(
        training_x, training_y, 
        test_size=test_size, 
        random_state=42, 
        stratify=training_y
    )
except ValueError as e:
    # If stratify fails (some classes have only 1 sample), split without stratify
    print(f"⚠️  Cannot stratify (some classes have too few samples)")
    print(f"   Using random split instead...")
    X_train, X_test, y_train, y_test = train_test_split(
        training_x, training_y, 
        test_size=test_size, 
        random_state=42
    )

print(f"\n✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Build model
print("\n🏗️ Building Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Train model
print("\n🚀 Training model...")
print("="*60)
model.fit(X_train, y_train)

print("\n✓ Training completed!")

# Evaluate
print("\n📊 Evaluating model...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"✓ Training Accuracy: {train_acc*100:.2f}%")
print(f"✓ Test Accuracy: {test_acc*100:.2f}%")

print("\n📋 Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test, zero_division=0))

# Test predictions
print("\n🧪 Testing predictions...")
test_sentences = [
    "halo",
    "apa itu phishing",
    "jelaskan tentang malware",
    "bagaimana cara membuat password yang aman",
    "apa itu ransomware"
]

for sentence in test_sentences:
    # Preprocess
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    
    # Bag of words
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    
    # Predict
    try:
        intent = model.predict([bag])[0]
        proba = model.predict_proba([bag])[0]
        class_idx = list(model.classes_).index(intent)
        confidence = proba[class_idx]
        
        # Get top 3
        top_3_idx = np.argsort(proba)[-3:][::-1]
        top_3 = [(model.classes_[i], proba[i]) for i in top_3_idx]
        
        print(f"\nInput: '{sentence}'")
        print(f"  → Intent: {intent}")
        print(f"  → Confidence: {confidence*100:.2f}%")
    except Exception as e:
        print(f"\nInput: '{sentence}'")
        print(f"  → Error: {e}")

# Feature importance
print("\n📊 Top 10 Most Important Words:")
feature_importance = model.feature_importances_
top_features_idx = np.argsort(feature_importance)[-10:][::-1]
for idx in top_features_idx:
    if idx < len(words):
        print(f"  {words[idx]}: {feature_importance[idx]:.4f}")

# Save model
print("\n💾 Saving model...")
pickle.dump(model, open('anonymous_model.pkl', 'wb'))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Save training info
model_info = {
    'total_documents': len(documents),
    'total_classes': len(classes),
    'total_words': len(words),
    'classes': classes,
    'training_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'model_type': 'RandomForestClassifier',
    'n_estimators': 200
}

with open('model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, indent=2, ensure_ascii=False)

print("\n✓ Model saved:")
print("  - anonymous_model.pkl")
print("  - words.pkl")
print("  - classes.pkl")
print("  - model_info.json")

# Plot accuracy comparison
try:
    print("\n📈 Creating accuracy plot...")
    plt.figure(figsize=(8, 6))
    accuracies = [train_acc * 100, test_acc * 100]
    labels = ['Training', 'Test']
    colors = ['#4CAF50', '#2196F3']
    plt.bar(labels, accuracies, color=colors, width=0.5)
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(0, 110)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_accuracy.png', dpi=300, bbox_inches='tight')
    print("✓ Accuracy plot saved: model_accuracy.png")
    plt.close()
except Exception as e:
    print(f"⚠️  Could not create plot: {e}")

print(f"""
╔═══════════════════════════════════════════════════════════╗
║              TRAINING COMPLETED SUCCESSFULLY!             ║
║  Training Accuracy: {train_acc*100:.2f}%                              ║
║  Test Accuracy: {test_acc*100:.2f}%                                  ║
║                                                           ║
║  Model Type: Random Forest (NO TENSORFLOW!)              ║
║                                                           ║
║  Next steps:                                              ║
║  1. Run: streamlit run app_sklearn.py                    ║
║  2. Test dengan berbagai input                            ║
║  3. Check confidence scores                               ║
╚═══════════════════════════════════════════════════════════╝
""")
"""
TRAINING SCRIPT - FIXED VERSION
Memperbaiki masalah low confidence & unknown intent
"""

import json
import numpy as np
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
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
║  ANONYMOUS CHATBOT - TRAINING (FIXED VERSION)            ║
║  Memperbaiki Low Confidence & Unknown Intent Issues      ║
╚═══════════════════════════════════════════════════════════╝
""")

# Initialize
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', ';', ':']

# Load intents
print("📂 Loading intents.json...")
try:
    with open('intents.json', encoding='utf-8') as file:
        intents = json.load(file)
    print(f"✓ Loaded {len(intents['intents'])} intents")
except Exception as e:
    print(f"❌ Error loading intents.json: {e}")
    exit(1)

# Preprocessing
print("\n📊 Preprocessing data...")
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize
        word_list = nltk.word_tokenize(pattern.lower())
        words.extend(word_list)
        # Add to documents
        documents.append((word_list, intent['tag']))
        # Add to classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean
print("🔧 Lemmatizing words...")
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

print(f"\n✓ Preprocessing completed:")
print(f"  - {len(documents)} documents")
print(f"  - {len(classes)} classes: {classes[:5]}...")
print(f"  - {len(words)} unique words")

# Create training data
print("\n🔧 Creating training data...")
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create bag of words
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training.append([bag, output_row])

# Shuffle and convert to array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split features and labels
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

print(f"✓ Training data shape: X{train_x.shape}, Y{train_y.shape}")

# Build model - SIMPLE & EFFECTIVE
print("\n🏗️ Building model...")
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile with SGD (lebih stabil untuk dataset kecil)
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("\n📋 Model Summary:")
model.summary()

# Train model
print(f"\n🚀 Training model...")
print("="*60)

history = model.fit(
    train_x, train_y,
    epochs=300,  # Lebih banyak epoch
    batch_size=5,  # Batch size lebih kecil
    verbose=1,
    validation_split=0.1
)

print("\n✓ Training completed!")

# Evaluate
print("\n📊 Evaluating model...")
train_loss, train_acc = model.evaluate(train_x, train_y, verbose=0)
print(f"✓ Training Accuracy: {train_acc*100:.2f}%")
print(f"✓ Training Loss: {train_loss:.4f}")

# Test predictions
print("\n🧪 Testing predictions...")
test_sentences = [
    "halo",
    "apa itu phishing",
    "jelaskan tentang malware",
    "bagaimana cara membuat password yang aman"
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
    res = model.predict(np.array([bag]), verbose=0)[0]
    top_idx = np.argmax(res)
    confidence = res[top_idx]
    intent = classes[top_idx]
    
    print(f"\nInput: '{sentence}'")
    print(f"  → Intent: {intent}")
    print(f"  → Confidence: {confidence*100:.2f}%")
    print(f"  → Top 3: {sorted(zip(classes, res), key=lambda x: x[1], reverse=True)[:3]}")

# Save model
print("\n💾 Saving model...")
model.save('anonymous_model.h5')
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Save training info
model_info = {
    'total_documents': len(documents),
    'total_classes': len(classes),
    'total_words': len(words),
    'classes': classes,
    'training_accuracy': float(train_acc),
    'training_loss': float(train_loss)
}

with open('model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, indent=2, ensure_ascii=False)

print("\n✓ Model saved:")
print("  - anonymous_model.h5")
print("  - words.pkl")
print("  - classes.pkl")
print("  - model_info.json")

# Plot training history
print("\n📈 Creating training history plot...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✓ Training history saved: training_history.png")

print(f"""
╔═══════════════════════════════════════════════════════════╗
║              TRAINING COMPLETED SUCCESSFULLY!             ║
║  Final Accuracy: {train_acc*100:.2f}%                                    ║
║                                                           ║
║  Next steps:                                              ║
║  1. Run: streamlit run app_improved.py                   ║
║  2. Test dengan berbagai input                            ║
║  3. Check confidence scores                               ║
╚═══════════════════════════════════════════════════════════╝
""")
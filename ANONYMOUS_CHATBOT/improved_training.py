import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocess_text

# Load dataset
with open('dataset_augmented.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Ekstrak data
texts = []
labels = []
for intent in data:
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(intent['intent'])

# Preprocessing
processed_texts = [preprocess_text(text) for text in texts]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Buat pipeline: TF-IDF + SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear', probability=True))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluasi
y_pred = pipeline.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.title('Confusion Matrix - Cybersecurity Chatbot')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_improved.png')
plt.show()

# Simpan model dan metadata
joblib.dump(pipeline, 'cybersec_chatbot_model.pkl')
print("\nModel saved as 'cybersec_chatbot_model.pkl'")

# Simpan intent responses untuk inference
intent_responses = {intent['intent']: intent['responses'] for intent in data}
joblib.dump(intent_responses, 'intent_responses.pkl')
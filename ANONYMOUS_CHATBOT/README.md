# 🛡️ Cybersecurity Chatbot (Machine Learning)

Chatbot berbasis Machine Learning untuk menjawab pertanyaan seputar cybersecurity
menggunakan pendekatan **Hybrid ML + Rule-Based + Context Awareness**.

## 🔍 Fitur
- Intent classification (Logistic Regression + TF-IDF)
- Dataset Bahasa Indonesia
- Context follow-up (contoh, lanjut, boleh)
- Evaluasi model (classification report & confusion matrix)
- UI berbasis Streamlit

## 📁 Struktur Project
├── app.py
├── cybersecurity_chatbot_advanced.py
├── intent_training_data.json
├── dataset_augmented.json
├── intent_model.pkl
├── vectorizer.pkl
└── README.md


## 🚀 Cara Menjalankan
1. Train model:
python cybersecurity_chatbot_advanced.py

2. Jalankan chatbot:
streamlit run app.py

🧠 Model

- TF-IDF Vectorizer
- Logistic Regression
- Akurasi ±65–75% (tanpa deep learning)

📊 Evaluasi
Menggunakan:
- Classification Report
- Confusion Matrix

👤 Author
Alessandro Farrel Gerrard Wijaya – Project Chatbot Cybersecurity Machine Learning
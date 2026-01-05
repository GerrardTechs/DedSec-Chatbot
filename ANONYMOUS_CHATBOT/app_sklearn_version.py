"""
ANONYMOUS CHATBOT - STREAMLIT APP (FINAL FIXED)
100% Compatible dengan format Anda
"""

import streamlit as st
import json
import numpy as np
import pickle
import random
import nltk
import os
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Page config
st.set_page_config(
    page_title="Anonymous Chatbot",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        animation: fadeIn 0.5s;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .bot-message {
        background: #f0f2f6;
        color: #1f1f1f;
        margin-right: 20%;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class AnonymousChatbotSklearn:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.model = None
        self.words = None
        self.classes = None
        self.intents = None
        
    @st.cache_resource
    def load_model_data(_self):
        """Load model dan data dengan caching"""
        try:
            model = pickle.load(open('anonymous_model.pkl', 'rb'))
            words = pickle.load(open('words.pkl', 'rb'))
            classes = pickle.load(open('classes.pkl', 'rb'))
            
            # Auto-detect intents file
            intent_files = [
                'dataset_augmented.json',
                'intent_training_data_expanded.json',
                'data_augmented.json',
                'intents.json'
            ]
            
            intents = None
            for filename in intent_files:
                try:
                    if os.path.exists(filename):
                        with open(filename, encoding='utf-8') as file:
                            data = json.load(file)
                            if 'intents' in data:
                                intents = data
                                break
                except:
                    continue
            
            if intents is None:
                st.error("❌ No intents file found!")
                return None, None, None, None
                
            return model, words, classes, intents
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info("💡 Pastikan Anda sudah menjalankan: python train_sklearn_final.py")
            return None, None, None, None
    
    def initialize(self):
        """Initialize model"""
        self.model, self.words, self.classes, self.intents = self.load_model_data()
        return self.model is not None
    
    def clean_up_sentence(self, sentence):
        """Preprocessing input"""
        sentence_words = nltk.word_tokenize(sentence.lower())
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words
    
    def bag_of_words(self, sentence):
        """Convert to bag of words"""
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)
    
    def predict_class(self, sentence):
        """Predict intent using sklearn"""
        bow = self.bag_of_words(sentence)
        
        try:
            # Predict dengan sklearn
            intent = self.model.predict([bow])[0]
            proba = self.model.predict_proba([bow])[0]
            
            # Get class index
            class_idx = list(self.model.classes_).index(intent)
            confidence = proba[class_idx]
            
            # Get top results
            ERROR_THRESHOLD = 0.15
            top_idx = np.argsort(proba)[::-1]
            
            return_list = []
            for idx in top_idx:
                if proba[idx] > ERROR_THRESHOLD:
                    return_list.append({
                        'intent': self.model.classes_[idx],
                        'probability': float(proba[idx])
                    })
            
            return return_list
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return []
    
    def get_response(self, intents_list):
        """Get response from intent"""
        if len(intents_list) == 0:
            return {
                'response': "🤔 Maaf, saya tidak yakin memahami pertanyaan Anda. Bisa dijelaskan lebih detail atau pilih topik dari Quick Actions?",
                'intent': 'unknown',
                'confidence': 0.0
            }
        
        tag = intents_list[0]['intent']
        confidence = intents_list[0]['probability']
        
        # Find matching intent (support both 'tag' and 'intent' keys)
        for i in self.intents['intents']:
            intent_tag = i.get('tag') or i.get('intent')
            if intent_tag == tag:
                responses = i.get('responses', [])
                if responses:
                    response = random.choice(responses)
                else:
                    response = f"Saya punya informasi tentang {tag}."
                
                return {
                    'response': response,
                    'intent': tag,
                    'confidence': confidence
                }
        
        return {
            'response': f"Maaf, saya tidak menemukan informasi tentang {tag}.",
            'intent': tag,
            'confidence': confidence
        }
    
    def chat(self, message):
        """Main chat function"""
        ints = self.predict_class(message)
        result = self.get_response(ints)
        return result

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'bot' not in st.session_state:
    st.session_state.bot = AnonymousChatbotSklearn()
    if not st.session_state.bot.initialize():
        st.stop()

# Header
st.markdown("""
<div class="main-header">
    <h1>🔒 ANONYMOUS CHATBOT</h1>
    <p style="font-size: 1.2rem; margin: 0;">Cybersecurity Information Assistant</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Quick Actions")
    st.markdown("---")
    
    st.markdown("### 🛡️ Security Topics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎣 Phishing", use_container_width=True, key="btn_phishing"):
            prompt = "Baik! Anda ingin belajar tentang Phishing. Silakan tanya:\n\n• Apa itu phishing?\n• Bagaimana cara kerja phishing?\n• Ciri-ciri email phishing?\n• Cara menghindari phishing?\n\nLangsung tanyakan saja! 😊"
            st.session_state.messages.append({
                'role': 'assistant',
                'content': prompt,
                'intent': 'phishing_guide',
                'confidence': 1.0
            })
            st.rerun()
        
        if st.button("🦠 Malware", use_container_width=True, key="btn_malware"):
            prompt = "Oke! Topik Malware. Silakan tanya:\n\n• Apa itu malware?\n• Jenis-jenis malware?\n• Bahaya malware?\n• Cara mencegah malware?\n\nApa yang ingin Anda ketahui? 🔍"
            st.session_state.messages.append({
                'role': 'assistant',
                'content': prompt,
                'intent': 'malware_guide',
                'confidence': 1.0
            })
            st.rerun()
        
        if st.button("🔐 Password", use_container_width=True, key="btn_password"):
            prompt = "Baik! Mari bahas Password Security. Tanyakan:\n\n• Cara membuat password kuat?\n• Apa itu password manager?\n• Tips keamanan password?\n• Pentingnya 2FA?\n\nSilakan tanya sesuai kebutuhan! 🔑"
            st.session_state.messages.append({
                'role': 'assistant',
                'content': prompt,
                'intent': 'password_guide',
                'confidence': 1.0
            })
            st.rerun()
    
    with col2:
        if st.button("💰 Ransomware", use_container_width=True, key="btn_ransomware"):
            prompt = "Ransomware! Topik penting. Tanyakan:\n\n• Apa itu ransomware?\n• Cara kerja ransomware?\n• Contoh serangan ransomware?\n• Cara mencegah ransomware?\n\nAyo tanya! 🚨"
            st.session_state.messages.append({
                'role': 'assistant',
                'content': prompt,
                'intent': 'ransomware_guide',
                'confidence': 1.0
            })
            st.rerun()
        
        if st.button("🌐 VPN", use_container_width=True, key="btn_vpn"):
            prompt = "VPN! Mari belajar. Silakan tanya:\n\n• Apa itu VPN?\n• Fungsi VPN?\n• Cara kerja VPN?\n• VPN yang aman?\n\nAda yang ingin ditanyakan? 🔒"
            st.session_state.messages.append({
                'role': 'assistant',
                'content': prompt,
                'intent': 'vpn_guide',
                'confidence': 1.0
            })
            st.rerun()
        
        if st.button("🔥 Firewall", use_container_width=True, key="btn_firewall"):
            prompt = "Firewall! Pertahanan penting. Tanyakan:\n\n• Apa itu firewall?\n• Fungsi firewall?\n• Jenis-jenis firewall?\n• Cara kerja firewall?\n\nSilakan! 🛡️"
            st.session_state.messages.append({
                'role': 'assistant',
                'content': prompt,
                'intent': 'firewall_guide',
                'confidence': 1.0
            })
            st.rerun()
    
    st.markdown("---")
    
    # Stats
    if len(st.session_state.messages) > 0:
        st.markdown("### 📊 Session Stats")
        st.metric("Total Messages", len(st.session_state.messages))
        
        confidences = [msg.get('confidence', 0) for msg in st.session_state.messages if msg['role'] == 'assistant']
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

# Main chat area
chat_container = st.container()

with chat_container:
    if len(st.session_state.messages) == 0:
        st.info("👋 Halo! Saya Anonymous, chatbot cybersecurity. Silakan tanya atau pilih topik dari sidebar!")
    
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 Anda:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = message.get('confidence', 0)
            intent = message.get('intent', 'unknown')
            
            if confidence >= 0.8:
                conf_class = "confidence-high"
                conf_emoji = "✅"
            elif confidence >= 0.5:
                conf_class = "confidence-medium"
                conf_emoji = "⚠️"
            else:
                conf_class = "confidence-low"
                conf_emoji = "❌"
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Anonymous:</strong><br>
                {message['content']}<br>
                <small style="opacity: 0.7;">
                    {conf_emoji} Intent: <code>{intent}</code> | 
                    Confidence: <span class="{conf_class}">{confidence*100:.1f}%</span>
                </small>
            </div>
            """, unsafe_allow_html=True)

# Input area
st.markdown("---")
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "💬 Tanyakan sesuatu...",
        key="user_input",
        placeholder="Contoh: Apa itu phishing?",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("📤 Send", use_container_width=True, type="primary")

if send_button and user_input:
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input
    })
    
    with st.spinner("🤔 Thinking..."):
        result = st.session_state.bot.chat(user_input)
    
    st.session_state.messages.append({
        'role': 'assistant',
        'content': result['response'],
        'intent': result['intent'],
        'confidence': result['confidence']
    })
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.6; font-size: 0.9rem;">
    🔒 Anonymous Chatbot | Sklearn Version | Powered by Random Forest
</div>
""", unsafe_allow_html=True)
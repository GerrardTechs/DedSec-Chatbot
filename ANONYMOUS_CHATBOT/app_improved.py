"""
ANONYMOUS CHATBOT - STREAMLIT APP (FIXED VERSION)
Cybersecurity Chatbot with Functional Buttons & Improved Training
"""

import streamlit as st
import json
import numpy as np
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
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
    .quick-action-btn {
        margin: 0.3rem;
    }
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

class AnonymousChatbot:
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
            model = load_model('anonymous_model.h5')
            words = pickle.load(open('words.pkl', 'rb'))
            classes = pickle.load(open('classes.pkl', 'rb'))
            with open('intents.json', encoding='utf-8') as file:
                intents = json.load(file)
            return model, words, classes, intents
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info("💡 Pastikan Anda sudah menjalankan: python train_model.py")
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
        """Predict intent"""
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]), verbose=0)[0]
        
        ERROR_THRESHOLD = 0.20  # Turunkan threshold untuk lebih sensitif
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list = []
        for r in results:
            return_list.append({
                'intent': self.classes[r[0]], 
                'probability': float(r[1])
            })
        return return_list
    
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
        
        for i in self.intents['intents']:
            if i['tag'] == tag:
                response = random.choice(i['responses'])
                return {
                    'response': response,
                    'intent': tag,
                    'confidence': confidence
                }
        
        return {
            'response': "Maaf, saya mengalami kendala. Silakan coba lagi.",
            'intent': 'error',
            'confidence': 0.0
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
    st.session_state.bot = AnonymousChatbot()
    if not st.session_state.bot.initialize():
        st.stop()

# Header
st.markdown("""
<div class="main-header">
    <h1>🔒 ANONYMOUS CHATBOT</h1>
    <p style="font-size: 1.2rem; margin: 0;">Cybersecurity Information Assistant</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">Powered by Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Quick Actions")
    st.markdown("---")
    
    # FUNCTIONAL BUTTONS dengan topik spesifik
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
    
    # More topics
    st.markdown("### 💡 More Topics")
    if st.button("🔒 Encryption", use_container_width=True):
        prompt = "Encryption! Tanyakan tentang:\n\n• Apa itu enkripsi?\n• Jenis enkripsi?\n• SSL/TLS?\n• End-to-end encryption?\n\nAyo! 🔐"
        st.session_state.messages.append({
            'role': 'assistant',
            'content': prompt,
            'intent': 'encryption_guide',
            'confidence': 1.0
        })
        st.rerun()
    
    if st.button("👤 Social Engineering", use_container_width=True):
        prompt = "Social Engineering! Tanyakan:\n\n• Apa itu social engineering?\n• Teknik-tekniknya?\n• Contoh serangan?\n• Cara mencegah?\n\nTanya aja! 🎭"
        st.session_state.messages.append({
            'role': 'assistant',
            'content': prompt,
            'intent': 'social_eng_guide',
            'confidence': 1.0
        })
        st.rerun()
    
    if st.button("⚡ DDoS Attack", use_container_width=True):
        prompt = "DDoS Attack! Silakan tanya:\n\n• Apa itu DDoS?\n• Cara kerja DDoS?\n• Dampak DDoS?\n• Cara mencegah?\n\nTanya yuk! 💥"
        st.session_state.messages.append({
            'role': 'assistant',
            'content': prompt,
            'intent': 'ddos_guide',
            'confidence': 1.0
        })
        st.rerun()
    
    st.markdown("---")
    
    # Stats
    if len(st.session_state.messages) > 0:
        st.markdown("### 📊 Session Stats")
        st.metric("Total Messages", len(st.session_state.messages))
        
        # Average confidence
        confidences = [msg.get('confidence', 0) for msg in st.session_state.messages if msg['role'] == 'assistant']
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Anonymous Chatbot v2.0**
        
        Chatbot berbasis Deep Learning untuk informasi Cybersecurity.
        
        **Features:**
        - Intent Classification
        - Confidence Scoring
        - Quick Action Buttons
        - Real-time Response
        
        **Tech Stack:**
        - TensorFlow/Keras
        - NLTK
        - Streamlit
        """)

# Main chat area
chat_container = st.container()

# Display messages
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
            
            # Confidence color
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

# Handle input
if send_button and user_input:
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input
    })
    
    # Get bot response
    with st.spinner("🤔 Thinking..."):
        result = st.session_state.bot.chat(user_input)
    
    # Add bot response
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
    🔒 Anonymous Chatbot | Cybersecurity AI Assistant | Powered by Deep Learning
</div>
""", unsafe_allow_html=True)
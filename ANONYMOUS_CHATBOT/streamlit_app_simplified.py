"""
═══════════════════════════════════════════════════════════════════════════════
🎨 SIMPLIFIED RAG CHATBOT - STREAMLIT APP (NO TORCH)
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
from datetime import datetime
import json

# Try to import RAG engine
try:
    from rag_engine_simplified import SimplifiedRAGEngine
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# 🎨 PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Cybersecurity RAG Chatbot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        padding: 1rem 0;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 📊 SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize session state"""
    if 'rag_engine' not in st.session_state:
        if RAG_AVAILABLE:
            try:
                with st.spinner("Loading RAG Engine..."):
                    st.session_state.rag_engine = SimplifiedRAGEngine()
                st.success("✅ RAG Engine loaded successfully!")
            except Exception as e:
                st.session_state.rag_engine = None
                st.error(f"❌ Error loading RAG Engine: {e}")
        else:
            st.session_state.rag_engine = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'conversation_count' not in st.session_state:
        st.session_state.conversation_count = 0

init_session_state()

# ═══════════════════════════════════════════════════════════════════════════
# 🎯 SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Cybersecurity RAG Chatbot")
    
    st.markdown("### 📊 System Info")
    st.info("""
    **Engine:** Simplified RAG
    **No heavy dependencies!**
    - ✅ sklearn (Intent Classification)
    - ✅ TF-IDF (Vector Search)
    - ✅ Template-based Responses
    """)
    
    st.markdown("---")
    
    # Stats
    st.markdown("### 📈 Stats")
    st.metric("Total Queries", st.session_state.conversation_count)
    st.metric("Messages", len(st.session_state.messages))
    
    # Clear Chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_count = 0
        st.rerun()
    
    # Export
    if st.session_state.messages:
        if st.button("💾 Export Chat", use_container_width=True):
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'messages': st.session_state.messages
            }
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Download JSON",
                json_str,
                f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Help
    with st.expander("❓ Help"):
        st.markdown("""
        **Contoh Pertanyaan:**
        - Apa itu phishing?
        - Cara mencegah malware?
        - Tips password yang aman?
        - Jelaskan tentang ransomware
        - Bedanya firewall sama antivirus?
        
        **Fitur:**
        - Intent classification
        - Context-aware responses
        - Fast & lightweight
        - No GPU required
        """)

# ═══════════════════════════════════════════════════════════════════════════
# 🎨 MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">🛡️ Cybersecurity RAG Chatbot</p>', unsafe_allow_html=True)

# Welcome message
if not st.session_state.messages:
    st.info("""
    👋 **Selamat datang!**
    
    Saya chatbot cybersecurity dengan RAG (Retrieval Augmented Generation).
    
    Tanyakan apa saja seputar keamanan siber!
    """)

# Check availability
if not RAG_AVAILABLE or not st.session_state.rag_engine:
    st.error("""
    ⚠️ **RAG Engine tidak tersedia!**
    
    **Troubleshooting:**
    1. Pastikan sudah training: `python intent_classifier.py`
    2. Pastikan vector store sudah dibuat: `python vector_store_simplified.py`
    3. Check error message di terminal
    """)
    st.stop()

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show debug info if available
        if msg["role"] == "assistant" and "metadata" in msg:
            with st.expander("🔍 Debug Info"):
                meta = msg["metadata"]
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Intent:** {meta.get('intent')}")
                    st.write(f"**Confidence:** {meta.get('confidence', 0):.2%}")
                with col2:
                    st.write(f"**Method:** {meta.get('method')}")
                    st.write(f"**Contexts:** {len(meta.get('contexts', []))}")

# ═══════════════════════════════════════════════════════════════════════════
# 💬 CHAT INPUT
# ═══════════════════════════════════════════════════════════════════════════

user_input = st.chat_input("Tanyakan seputar cybersecurity...")

if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.rag_engine.chat(user_input)
                response = result['response']
            except Exception as e:
                response = f"❌ Error: {str(e)}"
                result = {
                    'intent': 'error',
                    'confidence': 0.0,
                    'method': 'error',
                    'contexts': []
                }
        
        # Display response
        st.markdown(response)
        
        # Show debug info
        with st.expander("🔍 Debug Info"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Intent:** {result.get('intent')}")
                st.write(f"**Confidence:** {result.get('confidence', 0):.2%}")
            with col2:
                st.write(f"**Method:** {result.get('method')}")
                st.write(f"**Contexts:** {len(result.get('contexts', []))}")
    
    # Add to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "intent": result.get('intent'),
            "confidence": result.get('confidence'),
            "method": result.get('method'),
            "contexts": result.get('contexts', [])
        }
    })
    
    st.session_state.conversation_count += 1
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# 🚀 QUICK EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════

if not st.session_state.messages:
    st.markdown("---")
    st.markdown("### 🚀 Contoh Pertanyaan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    examples = {
        "💡 Phishing": "apa itu phishing dan cara mencegahnya?",
        "💡 Malware": "jelaskan tentang malware",
        "💡 Password": "tips password yang aman",
        "💡 Firewall": "apa fungsi firewall?"
    }
    
    for i, (col, (label, query)) in enumerate(zip([col1, col2, col3, col4], examples.items())):
        with col:
            if st.button(label, use_container_width=True, key=f"ex_{i}"):
                st.session_state.pending_query = query
                st.rerun()

# Handle pending query
if hasattr(st.session_state, 'pending_query'):
    query = st.session_state.pending_query
    delattr(st.session_state, 'pending_query')
    
    # Add to messages and process
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        result = st.session_state.rag_engine.chat(query)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['response'],
            "timestamp": datetime.now().isoformat(),
            "metadata": result
        })
        st.session_state.conversation_count += 1
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
    
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# 📌 FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🛡️ <b>Cybersecurity RAG Chatbot</b> - Simplified Version</p>
    <p>Fast • Lightweight • No GPU Required</p>
</div>
""", unsafe_allow_html=True)

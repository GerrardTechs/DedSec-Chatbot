"""
═══════════════════════════════════════════════════════════════════════════════
🔧 RAG CHATBOT - CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════
Central configuration for all RAG components
"""

import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# 📁 PATHS
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
OUTPUTS_DIR = BASE_DIR / 'outputs'

# Create directories
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Data files
INTENT_TRAINING_DATA = 'intent_training_data_expanded.json'
RESPONSE_DATABASE = 'dataset_augmented.json'

# Model paths
INTENT_MODEL_DIR = MODELS_DIR / 'intent_classifier'
VECTOR_STORE_DIR = MODELS_DIR / 'vector_store'
LLM_MODEL_DIR = MODELS_DIR / 'llm'

# Output files
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / 'confusion_matrix_rag.png'
EVALUATION_REPORT_PATH = OUTPUTS_DIR / 'evaluation_report.html'
CLASSIFICATION_REPORT_PATH = OUTPUTS_DIR / 'classification_report.txt'

# ═══════════════════════════════════════════════════════════════════════════
# 🤖 MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

# Intent Classifier
INTENT_CONFIG = {
    'test_size': 0.2,  # 80/20 split
    'random_state': 42,
    'use_ensemble': True,
    'models': ['logistic_regression', 'random_forest', 'svc'],
    'tfidf_params': {
        'ngram_range': (1, 3),
        'max_features': 2000,
        'min_df': 2,
        'max_df': 0.8
    }
}

# Embedding Model
EMBEDDING_CONFIG = {
    'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'device': 'cpu',  # Change to 'cuda' if GPU available
    'batch_size': 32,
    'normalize_embeddings': True
}

# FAISS Vector Store
VECTOR_STORE_CONFIG = {
    'index_type': 'IndexFlatL2',  # or 'IndexIVFFlat' for large data
    'dimension': 384,  # MiniLM embedding dimension
    'nlist': 100,  # for IVF index
    'nprobe': 10,  # search clusters
    'top_k': 3  # retrieve top 3 contexts
}

# LLM Configuration
LLM_CONFIG = {
    'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
    'device': 'cpu',  # Change to 'cuda' if GPU available
    'quantization': '4bit',  # Options: None, '4bit', '8bit'
    'max_new_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.9,
    'repetition_penalty': 1.1,
    'do_sample': True
}

# ═══════════════════════════════════════════════════════════════════════════
# 🔍 RAG CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

RAG_CONFIG = {
    'use_intent_classification': True,
    'use_vector_search': True,
    'use_llm_generation': True,
    'fallback_to_templates': True,
    
    # Confidence thresholds
    'intent_confidence_high': 0.75,
    'intent_confidence_medium': 0.50,
    'intent_confidence_low': 0.30,
    
    # Context management
    'max_context_length': 2048,
    'context_window_size': 5,  # remember last 5 turns
    
    # Response generation
    'use_streaming': False,
    'max_response_length': 500
}

# ═══════════════════════════════════════════════════════════════════════════
# 🌐 API CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': False,  # Set True for development
    'workers': 1,
    
    # CORS
    'allow_origins': ['*'],
    'allow_credentials': True,
    'allow_methods': ['*'],
    'allow_headers': ['*'],
    
    # Rate limiting
    'rate_limit_enabled': True,
    'rate_limit_calls': 60,
    'rate_limit_period': 60  # seconds
}

# ═══════════════════════════════════════════════════════════════════════════
# 🎨 UI CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

UI_CONFIG = {
    'title': 'Cybersecurity RAG Chatbot',
    'icon': '🛡️',
    'layout': 'wide',
    'theme': 'light',
    
    # Features
    'show_debug_info': True,
    'show_confidence': True,
    'show_context': True,
    'show_evaluation': True,
    
    # Chat
    'max_messages_display': 50,
    'enable_export': True,
    'enable_feedback': True
}

# ═══════════════════════════════════════════════════════════════════════════
# 📊 EVALUATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

EVAL_CONFIG = {
    # Metrics
    'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
    'average_methods': ['micro', 'macro', 'weighted'],
    
    # Thresholds
    'min_accuracy': 0.85,
    'min_f1_score': 0.80,
    'min_precision': 0.80,
    'min_recall': 0.75,
    
    # Visualization
    'confusion_matrix_figsize': (15, 12),
    'plot_style': 'seaborn',
    'save_plots': True
}

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 PREPROCESSING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PREPROCESS_CONFIG = {
    'remove_stopwords': True,
    'apply_stemming': True,
    'normalize_typos': True,
    'min_word_length': 2,
    
    # Custom normalization dictionary
    'typo_corrections': {
        'phising': 'phishing',
        'pishing': 'phishing',
        'fisyhing': 'phishing',
        'malwer': 'malware',
        'malwere': 'malware',
        'firewol': 'firewall',
        'ransomwer': 'ransomware',
        'ddoss': 'ddos'
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# 🚀 DEPLOYMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DEPLOYMENT_CONFIG = {
    'environment': 'development',  # 'development' or 'production'
    'debug': True,
    'log_level': 'INFO',
    'log_file': 'rag_chatbot.log',
    
    # Performance
    'enable_caching': True,
    'cache_size': 1000,
    'enable_profiling': False
}

# ═══════════════════════════════════════════════════════════════════════════
# 💾 CACHE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CACHE_CONFIG = {
    'enabled': True,
    'backend': 'memory',  # 'memory' or 'redis'
    'ttl': 3600,  # seconds
    'max_size': 1000,
    
    # Redis (if used)
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0
}

# ═══════════════════════════════════════════════════════════════════════════
# 📝 LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'filename': 'rag_chatbot.log',
            'mode': 'a'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# 🎯 PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATES = {
    'system_prompt': """Anda adalah asisten ahli keamanan siber yang membantu pengguna memahami topik cybersecurity dalam bahasa Indonesia. 

Tugas Anda:
1. Memberikan penjelasan yang akurat dan mudah dipahami
2. Menggunakan bahasa Indonesia yang baik dan benar
3. Memberikan contoh konkret jika relevan
4. Fokus pada informasi yang bermanfaat dan praktis

Gaya komunikasi:
- Ramah dan profesional
- Jelas dan terstruktur
- Tidak menggunakan jargon berlebihan
- Memberikan actionable advice

Konteks yang relevan akan diberikan untuk membantu Anda menjawab dengan lebih akurat.""",
    
    'user_prompt_template': """Pertanyaan: {query}

Konteks relevan:
{context}

Berdasarkan konteks di atas, berikan penjelasan yang lengkap dan mudah dipahami tentang pertanyaan pengguna. Jika konteks tidak cukup, gunakan pengetahuan umum Anda tentang cybersecurity.""",
    
    'comparison_prompt_template': """Pertanyaan: {query}

Konteks tentang {concept1}:
{context1}

Konteks tentang {concept2}:
{context2}

Jelaskan perbedaan antara kedua konsep tersebut dengan jelas dan terstruktur."""
}

# ═══════════════════════════════════════════════════════════════════════════
# 🔐 SECURITY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SECURITY_CONFIG = {
    'enable_input_validation': True,
    'max_query_length': 500,
    'allowed_characters': 'alphanumeric_punctuation',
    'sanitize_output': True,
    
    # Rate limiting per IP
    'rate_limit_per_ip': {
        'enabled': True,
        'max_requests': 100,
        'time_window': 3600  # 1 hour
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# ✅ VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required files
    if not Path(INTENT_TRAINING_DATA).exists():
        errors.append(f"Training data not found: {INTENT_TRAINING_DATA}")
    
    if not Path(RESPONSE_DATABASE).exists():
        errors.append(f"Response database not found: {RESPONSE_DATABASE}")
    
    # Check threshold values
    if not 0 < INTENT_CONFIG['test_size'] < 1:
        errors.append("test_size must be between 0 and 1")
    
    if EVAL_CONFIG['min_accuracy'] > 1 or EVAL_CONFIG['min_accuracy'] < 0:
        errors.append("min_accuracy must be between 0 and 1")
    
    return errors

# ═══════════════════════════════════════════════════════════════════════════
# 🎯 EXPORT CONFIG
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    'INTENT_CONFIG',
    'EMBEDDING_CONFIG',
    'VECTOR_STORE_CONFIG',
    'LLM_CONFIG',
    'RAG_CONFIG',
    'API_CONFIG',
    'UI_CONFIG',
    'EVAL_CONFIG',
    'PREPROCESS_CONFIG',
    'DEPLOYMENT_CONFIG',
    'CACHE_CONFIG',
    'LOGGING_CONFIG',
    'PROMPT_TEMPLATES',
    'SECURITY_CONFIG',
    'validate_config'
]

if __name__ == '__main__':
    errors = validate_config()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration validated successfully!")

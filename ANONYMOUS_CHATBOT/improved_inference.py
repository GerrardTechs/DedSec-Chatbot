import joblib
import random
from pathlib import Path
from preprocessing import preprocess_text

# BASE DIRECTORY (aman di Streamlit Cloud)
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "cybersec_chatbot_model.pkl"
RESP_PATH = BASE_DIR / "intent_responses.pkl"

_model = None
_intent_responses = None

def load_resources():
    global _model, _intent_responses
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _intent_responses = joblib.load(RESP_PATH)
    return _model, _intent_responses

def get_response(user_input):
    model, intent_responses = load_resources()

    processed_input = preprocess_text(user_input)
    intent = model.predict([processed_input])[0]
    confidence = max(model.predict_proba([processed_input])[0])

    response = random.choice(intent_responses[intent])
    return response, intent, confidence


# CLI test (AMAN)
if __name__ == "__main__":
    print("Anonymous: Halo! Saya chatbot cybersecurity.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["keluar", "exit", "quit"]:
            break
        res, intent, conf = get_response(user_input)
        print(f"Anonymous ({intent}, {conf:.2f}): {res}\n")

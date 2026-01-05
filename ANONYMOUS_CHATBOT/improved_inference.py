import joblib
import random
from preprocessing import preprocess_text

# Load model dan respons
model = joblib.load('cybersec_chatbot_model.pkl')
intent_responses = joblib.load('intent_responses.pkl')

def get_response(user_input):
    processed_input = preprocess_input = preprocess_text(user_input)
    intent = model.predict([processed_input])[0]
    confidence = max(model.predict_proba([processed_input])[0])
    
    response = random.choice(intent_responses[intent])
    return response, intent, confidence

# CLI Demo
if __name__ == "__main__":
    print("Anonymous: Halo! Saya chatbot cybersecurity. Ketik 'keluar' untuk berhenti.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['keluar', 'exit', 'quit']:
            print("Anonymous: Sampai jumpa!")
            break
        response, intent, conf = get_response(user_input)
        print(f"Anonymous ({intent}, {conf:.2f}): {response}\n")
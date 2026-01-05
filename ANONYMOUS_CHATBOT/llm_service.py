"""
═══════════════════════════════════════════════════════════════════════════════
🤖 LLM SERVICE - QWEN 2.5 INTEGRATION
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

from rag_config import LLM_CONFIG, PROMPT_TEMPLATES

class LLMService:
    def __init__(self):
        print(f"\n{'='*80}")
        print("🤖 INITIALIZING LLM SERVICE")
        print('='*80)
        
        model_name = LLM_CONFIG['model_name']
        device = LLM_CONFIG['device']
        
        print(f"📦 Loading model: {model_name}")
        print(f"🖥️  Device: {device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with quantization if specified
        if LLM_CONFIG['quantization'] == '4bit':
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map='auto'
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                device_map='auto'
            )
        
        print(f"✅ LLM loaded successfully!")
    
    def generate(self, query: str, contexts: List[Dict]) -> str:
        """Generate response with RAG contexts"""
        
        # Build context text
        context_text = "\n\n".join([ctx['text'] for ctx in contexts])
        
        # Build prompt
        system_prompt = PROMPT_TEMPLATES['system_prompt']
        user_prompt = PROMPT_TEMPLATES['user_prompt_template'].format(
            query=query,
            context=context_text
        )
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors='pt').to(self.model.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=LLM_CONFIG['max_new_tokens'],
            temperature=LLM_CONFIG['temperature'],
            top_p=LLM_CONFIG['top_p'],
            repetition_penalty=LLM_CONFIG['repetition_penalty'],
            do_sample=LLM_CONFIG['do_sample']
        )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        if "Jawaban:" in response:
            response = response.split("Jawaban:")[-1].strip()
        elif full_prompt in response:
            response = response.replace(full_prompt, "").strip()
        
        return response

if __name__ == '__main__':
    llm = LLMService()
    
    contexts = [{'text': 'Phishing adalah teknik penipuan siber.'}]
    response = llm.generate("apa itu phishing", contexts)
    print(f"Response: {response}")
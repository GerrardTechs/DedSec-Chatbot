"""
Simple test to isolate load_data issue
"""

import json

print("="*80)
print("🧪 TESTING DATA LOADING")
print("="*80)

# Test loading
filename = 'intent_training_data_expanded.json'

print(f"\n📂 Loading: {filename}")

with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ Loaded successfully!")
print(f"📊 Data type: {type(data)}")

# Test conversion logic
texts = []
labels = []

if isinstance(data, dict):
    print(f"\n📊 Processing dict with {len(data)} keys...")
    
    for intent, samples in data.items():
        print(f"\n   Intent: {intent}")
        print(f"   Samples type: {type(samples)}")
        
        if isinstance(samples, list):
            print(f"   Samples count: {len(samples)}")
            
            for sample in samples:
                print(f"   Sample type: {type(sample)}")
                
                if isinstance(sample, str):
                    texts.append(sample)
                    labels.append(intent)
                    print(f"   ✅ Added string: {sample[:50]}")
                elif isinstance(sample, dict) and 'text' in sample:
                    texts.append(sample['text'])
                    labels.append(intent)
                    print(f"   ✅ Added dict text: {sample['text'][:50]}")
                else:
                    print(f"   ❌ Skipped: {type(sample)}")
                
                # Only show first 2 samples per intent
                if len([l for l in labels if l == intent]) >= 2:
                    print(f"   ... (showing first 2 only)")
                    break

print(f"\n{'='*80}")
print(f"📊 RESULTS")
print('='*80)
print(f"Total texts: {len(texts)}")
print(f"Total labels: {len(labels)}")
print(f"Unique intents: {len(set(labels))}")

if len(texts) > 0:
    print(f"\nFirst 5 texts:")
    for i, (text, label) in enumerate(zip(texts[:5], labels[:5])):
        print(f"   {i+1}. [{label}] {text}")
else:
    print("\n❌ NO TEXTS LOADED!")
    print("\nDEBUG: Check data structure")
    print(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    if isinstance(data, dict) and len(data) > 0:
        first_key = list(data.keys())[0]
        first_value = data[first_key]
        print(f"\nFirst key: {first_key}")
        print(f"First value type: {type(first_value)}")
        print(f"First value: {first_value}")
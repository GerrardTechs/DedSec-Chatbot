"""
Debug script to check training data file
"""

import json
import os

print("="*80)
print("🔍 DEBUGGING TRAINING DATA FILE")
print("="*80)

# Check current directory
print(f"\n📁 Current directory: {os.getcwd()}")

# Check if file exists
filename = 'intent_training_data_expanded.json'
print(f"\n🔍 Checking for: {filename}")

if os.path.exists(filename):
    print(f"✅ File exists!")
    
    # Check file size
    size = os.path.getsize(filename)
    print(f"📊 File size: {size} bytes")
    
    if size == 0:
        print("❌ ERROR: File is empty!")
    else:
        # Try to load
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ JSON loaded successfully!")
            print(f"📊 Data type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"📊 Number of keys: {len(data)}")
                print(f"📊 Keys (first 10): {list(data.keys())[:10]}")
                
                # Count samples
                total_samples = 0
                for key, value in data.items():
                    if isinstance(value, list):
                        total_samples += len(value)
                        print(f"   - {key}: {len(value)} samples")
                
                print(f"\n✅ Total samples: {total_samples}")
            
            elif isinstance(data, list):
                print(f"📊 Number of items: {len(data)}")
                if len(data) > 0:
                    print(f"📊 First item: {data[0]}")
            
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON format!")
            print(f"   Error: {e}")
        except Exception as e:
            print(f"❌ ERROR: {e}")
else:
    print(f"❌ File NOT found!")
    print(f"\n📁 Files in current directory:")
    for f in os.listdir('.'):
        if f.endswith('.json'):
            print(f"   - {f}")

print("\n" + "="*80)
print("🔍 CHECKING ALTERNATIVE FILENAMES")
print("="*80)

alternatives = [
    'intent_training_data.json',
    'intent_training_data_PROPER.json',
    'training_data.json',
    'intents.json'
]

for alt in alternatives:
    if os.path.exists(alt):
        size = os.path.getsize(alt)
        print(f"✅ Found: {alt} ({size} bytes)")

print("\n" + "="*80)
print("💡 SOLUTION")
print("="*80)

if not os.path.exists(filename):
    print("""
ERROR: File 'intent_training_data_expanded.json' tidak ditemukan!

SOLUSI:
1. Download file 'intent_training_data_PROPER.json' dari outputs
2. Copy ke folder: D:\\PROJEKAN\\ANONYMOUS_CHATBOT
3. Rename: ren intent_training_data_PROPER.json intent_training_data_expanded.json
4. Run ulang: python intent_classifier.py
    """)
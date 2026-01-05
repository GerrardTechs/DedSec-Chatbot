"""
SIMPLE JSON CONVERTER
Convert any JSON structure to simple format
"""

import json
import shutil
import os
from datetime import datetime

print("="*70)
print(" SIMPLE JSON CONVERTER")
print("="*70)

def extract_all_values(obj, result_list):
    """Recursively extract all string values from nested structure"""
    if isinstance(obj, dict):
        for value in obj.values():
            extract_all_values(value, result_list)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                result_list.append(item)
            else:
                extract_all_values(item, result_list)
    elif isinstance(obj, str):
        result_list.append(obj)

def convert_to_simple_format(data):
    """Convert any structure to simple intent format"""
    result = []
    
    for intent_name, content in data.items():
        # Extract all strings from nested structure
        all_values = []
        extract_all_values(content, all_values)
        
        # Split into patterns and responses
        # Patterns are usually shorter (queries), responses longer (answers)
        patterns = []
        responses = []
        
        for value in all_values:
            # Simple heuristic: shorter than 100 chars = pattern, longer = response
            if len(value) < 100 and ('?' in value or 'apa' in value.lower() or len(value.split()) < 10):
                patterns.append(value)
            else:
                responses.append(value)
        
        # If no patterns found, create default ones
        if not patterns:
            patterns = [
                f"apa itu {intent_name}",
                f"jelaskan {intent_name}",
                f"{intent_name} itu apa",
                f"info {intent_name}"
            ]
        
        # If no responses found, create default
        if not responses:
            responses = [f"Ini adalah informasi tentang {intent_name}."]
        
        # Limit responses to avoid bloat
        if len(responses) > 10:
            responses = responses[:10]
        
        intent_obj = {
            "intent": intent_name,
            "patterns": patterns,
            "responses": responses,
            "context": ""
        }
        
        result.append(intent_obj)
    
    return result

# Find JSON files
json_files = [f for f in os.listdir('.') if f.endswith('.json') and 'backup' not in f and 'converted' not in f]

if not json_files:
    print("No JSON files found!")
    exit(1)

print(f"\nFound {len(json_files)} JSON file(s)")

for filename in json_files:
    print(f"\n{'='*70}")
    print(f"Processing: {filename}")
    print('='*70)
    
    try:
        # Load
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("✓ Loaded")
        
        # Check if already in correct format
        if isinstance(data, dict) and 'intents' in data:
            print("✓ Already in correct format, skipping")
            continue
        
        # Convert
        print("Converting...")
        
        if isinstance(data, dict):
            simple_format = convert_to_simple_format(data)
        else:
            print("Unknown format, skipping")
            continue
        
        # Wrap
        final_data = {"intents": simple_format}
        
        # Stats
        total_patterns = sum(len(i['patterns']) for i in simple_format)
        total_responses = sum(len(i['responses']) for i in simple_format)
        
        print(f"✓ Converted {len(simple_format)} intents")
        print(f"✓ {total_patterns} patterns")
        print(f"✓ {total_responses} responses")
        
        # Sample
        print("\nSample:")
        for intent in simple_format[:3]:
            print(f"  • {intent['intent']}: {len(intent['patterns'])} patterns")
        
        # Backup
        backup = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filename, backup)
        print(f"\n✓ Backup: {backup}")
        
        # Save
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved: {filename}")
        
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "="*70)
print(" DONE!")
print("="*70)
print("\nNext: python train_sklearn_version.py")
print("="*70)
#!/usr/bin/env python3
"""
Test script for large JSON processing optimization
"""

import json
import time
import requests

# Test data - simplified version of the large JSON
test_data = [
    {
        "uri": "3392171",
        "question": "do braces hurt????",
        "context": "pain?\nhard to talk?",
        "answers": [
            "yes yes yes. But not horribly painful. you get used to them and they become easier to talk with. the pain is only when they get tightened and until you get used to them. They give you wax to put around the spots that hurt you when your tongue rubs against the brackets and you won't feel the pain.",
            "They hurt for a bit when you first get them.They feel tight.Then, they settle down. However, they hurt each time you get them adjusted for a few days afterward. But,dont worry,you'll get used to the pain."
        ],
        "labelled_summaries": {
            "EXPERIENCE_SUMMARY": "The various experiences with braces are shared, with some individuals describing initial pain and discomfort, especially during adjustments or the first few days after getting them. However, a common theme emerges that people tend to get used to the discomfort over time."
        }
    },
    {
        "uri": "1100363",
        "question": "i need sum serious answer??",
        "context": "ok .. i play sports all muh life .. yeah.. n i never lose weight ??? wut da hell do u think is the problem????",
        "answers": [
            "That might be because you're eating more energy than what you are spending. Other reason is that you must remember that when you do exercise your muscle get fit and maybe get bigger so it will be showed as weight, and it won't be unhealthy weight (from fat)",
            "its probable just ur matabolizm if u eaty alot of carbs that could be the problem."
        ]
    }
]

def test_regular_endpoint():
    """Test the regular /summarize endpoint"""
    print("Testing regular /summarize endpoint...")
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/summarize",
            json={
                "text": json.dumps(test_data),
                "mode": "both"
            },
            headers={"Authorization": "Bearer test_token"},
            timeout=30
        )
        
        if response.status_code == 200:
            print(" Regular endpoint successful")
            data = response.json()
            print(f"   Patient summary length: {len(data.get('patient_summary', ''))}")
            print(f"   Clinical summary length: {len(data.get('clinician_summary', ''))}")
        else:
            print(f" Regular endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f" Regular endpoint error: {e}")

def test_json_endpoint():
    """Test the /summarize/json endpoint"""
    print("\nTesting /summarize/json endpoint...")
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/summarize/json",
            json=test_data,
            headers={"Authorization": "Bearer test_token"},
            timeout=30
        )
        
        if response.status_code == 200:
            print("JSON endpoint successful")
            data = response.json()
            print(f"   Patient summary length: {len(data.get('patient_summary', ''))}")
            print(f"   Clinical summary length: {len(data.get('clinician_summary', ''))}")
            print(f"   Processing info: {data.get('processing_info', {})}")
        else:
            print(f" JSON endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f" JSON endpoint error: {e}")

def test_large_json_endpoint():
    """Test the /summarize/large-json endpoint"""
    print("\nTesting /summarize/large-json endpoint...")
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/summarize/large-json",
            json=test_data,
            headers={"Authorization": "Bearer test_token"},
            timeout=30
        )
        
        if response.status_code == 200:
            print("Large JSON endpoint successful")
            data = response.json()
            print(f"   Patient summary length: {len(data.get('patient_summary', ''))}")
            print(f"   Clinical summary length: {len(data.get('clinician_summary', ''))}")
            print(f"   Processing info: {data.get('processing_info', {})}")
        else:
            print(f"Large JSON endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"Large JSON endpoint error: {e}")

if __name__ == "__main__":
    print("Testing Large JSON Processing Optimization")
    print("=" * 50)
    
    # Wait for backend to be ready
    print("Waiting for backend to be ready...")
    time.sleep(3)
    
    # Test all endpoints
    test_regular_endpoint()
    test_json_endpoint()
    test_large_json_endpoint()
    
    print("\n Testing complete!")

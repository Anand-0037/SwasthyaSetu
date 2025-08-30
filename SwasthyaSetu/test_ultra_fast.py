#!/usr/bin/env python3
"""
Test script for ultra-fast JSON processing
"""

import json
import time
import requests

# Test data - your large JSON input
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
    }
]

def test_ultra_fast_endpoint():
    """Test the ultra-fast endpoint"""
    print("🚀 Testing Ultra-Fast JSON Processing")
    print("=" * 50)
    
    try:
        start_time = time.time()
        
        response = requests.post(
            "http://127.0.0.1:8000/summarize/ultra-fast",
            json=test_data,
            headers={"Authorization": "Bearer test_token"},
            timeout=10
        )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            print("✅ Ultra-fast endpoint successful!")
            data = response.json()
            
            print(f"\n📊 Performance Results:")
            print(f"   Total request time: {total_time:.3f} seconds")
            print(f"   Processing time: {data.get('processing_info', {}).get('processing_time_seconds', 'N/A')} seconds")
            print(f"   Performance rating: {data.get('processing_info', {}).get('performance', 'N/A')}")
            
            print(f"\n📝 Generated Summaries:")
            print(f"   Patient Summary: {data.get('patient_summary', '')[:100]}...")
            print(f"   Clinical Summary: {data.get('clinician_summary', '')[:100]}...")
            
            print(f"\n📈 Quality Metrics:")
            print(f"   Faithfulness Score: {data.get('faithfulness_score', 'N/A')}")
            print(f"   Message: {data.get('message', 'N/A')}")
            
        else:
            print(f"❌ Ultra-fast endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Ultra-fast endpoint error: {e}")

def test_regular_endpoint_comparison():
    """Test regular endpoint for comparison"""
    print("\n🔄 Testing Regular Endpoint for Comparison")
    print("=" * 50)
    
    try:
        start_time = time.time()
        
        response = requests.post(
            "http://127.0.0.1:8000/summarize",
            json={
                "text": json.dumps(test_data),
                "mode": "both"
            },
            headers={"Authorization": "Bearer test_token"},
            timeout=30
        )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            print("✅ Regular endpoint successful!")
            print(f"   Total request time: {total_time:.3f} seconds")
        else:
            print(f"❌ Regular endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Regular endpoint error: {e}")

if __name__ == "__main__":
    print("🏥 Medical Summarizer - Ultra-Fast Processing Test")
    print("=" * 60)
    
    # Wait for backend to be ready
    print("Waiting for backend to be ready...")
    time.sleep(3)
    
    # Test ultra-fast endpoint
    test_ultra_fast_endpoint()
    
    # Test regular endpoint for comparison
    test_regular_endpoint_comparison()
    
    print("\n✨ Testing complete!")
    print("\n💡 Use the /summarize/ultra-fast endpoint for instant results!")

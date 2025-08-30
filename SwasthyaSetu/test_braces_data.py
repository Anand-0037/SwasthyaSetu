#!/usr/bin/env python3
"""
Test script for SwasthyaSetu endpoints with braces data
"""

import requests
import json
import time

# Your braces data
braces_data = [
    {
        "uri": "3392171",
        "question": "do braces hurt????",
        "context": "pain? hard to talk?",
        "answers": [
            "yes yes yes. But not horribly painful. you get used to them and they become easier to talk with. the pain is only when they get tightened and until you get used to them. They give you wax to put around the spots that hurt you when your tongue rubs against the brackets and you won't feel the pain.",
            "They hurt for a bit when you first get them.They feel tight.Then, they settle down. However, they hurt each time you get them adjusted for a few days afterward. But,dont worry,you'll get used to the pain.",
            "yes yes",
            "they hurt when u first get them but then u get used 2 them really  easily. sometimes u like drool or spit or something but unless u have a pallete expander u can talk normal. if u have 1 ur \"K's\" get all screwed up.",
            "Yes THEy HuRt WhEn YoU fIrSt GeT them but just for a few days...If you eat something hard and it gets stuck in them it mite hurt...and also after youy get them tightened..but they do not make it hard to talk, only retainers do that!!!",
            "at the begining you feel really weird and it hurts a little bit but then you get used to them\nis not hard to talk at all you can talk normally",
            "I had them...they hurt for a bit when you first get them...feel tight.  then, they settle down.  However, they hurt each time you get them adjusted for a few days afterward.",
            "I just got my braces a few days ago. They felt like they were loose for the first two days, and it was really hard to eat. I got over speaking difficulties quickly, although the inside of your mouth gets kind of scratched and torn- don't fret, it heals quickly enough. After that you get calouses, so it doesn't hurt anymore. I think the pain felt depends on how old you are when you get them, what kind of treatment you are getting, and how severe your bite is. Lately, I've forgotten I'd had them on. You get used to them.",
            "Intially they are painful...also depending on your age and the condition your teeth are in. I have had my for almost 2 1/2 years and my first year was painful. Now I forget I have them...they are coming off soon and I can't wait! I recommend Aleve, its really helpful in relieving the pain."
        ]
    }
]

def test_swasthyasetu_t5_endpoint():
    """Test the SwasthyaSetu T5 endpoint with braces data"""
    print("🏥 Testing SwasthyaSetu T5 Endpoint with Braces Data")
    print("=" * 60)
    
    print(f"📊 Data Size: {len(json.dumps(braces_data))} characters")
    
    try:
        start_time = time.time()
        
        print("\n🔄 Sending request to SwasthyaSetu T5 endpoint...")
        response = requests.post(
            "http://127.0.0.1:8000/summarize/swasthyasetu-t5",
            json=braces_data,
            headers={"Authorization": "Bearer test_token"},
            timeout=60
        )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SwasthyaSetu T5 endpoint successful!")
            print(f"   Processing time: {total_time:.3f} seconds")
            print(f"   Model used: {data.get('processing_info', {}).get('model_used', 'N/A')}")
            
            print(f"\n📝 Generated Summaries:")
            print(f"   Patient Summary: {data.get('patient_summary', '')}")
            print(f"\n   Clinical Summary: {data.get('clinician_summary', '')}")
            
            print(f"\n📈 Quality Metrics:")
            print(f"   Faithfulness Score: {data.get('faithfulness_score', 'N/A')}")
            print(f"   Message: {data.get('message', 'N/A')}")
            
        else:
            print(f"❌ SwasthyaSetu T5 endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ SwasthyaSetu T5 endpoint error: {e}")

def test_ultra_fast_endpoint():
    """Test ultra-fast endpoint with braces data"""
    print("\n🔄 Testing Ultra-Fast Endpoint with Braces Data")
    print("=" * 50)
    
    try:
        start_time = time.time()
        
        response = requests.post(
            "http://127.0.0.1:8000/summarize/ultra-fast",
            json=braces_data,
            headers={"Authorization": "Bearer test_token"},
            timeout=30
        )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Ultra-fast endpoint successful!")
            print(f"   Processing time: {total_time:.3f} seconds")
            print(f"   Patient Summary: {data.get('patient_summary', '')[:100]}...")
            print(f"   Clinical Summary: {data.get('clinician_summary', '')[:100]}...")
        else:
            print(f"❌ Ultra-fast endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Ultra-fast endpoint error: {e}")

def test_model_info():
    """Test model info endpoint"""
    print("\n📊 Testing Model Info Endpoint")
    print("=" * 40)
    
    try:
        response = requests.get("http://127.0.0.1:8000/model/info")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved successfully!")
            print(f"   Project: {data.get('project_name', 'N/A')}")
            print(f"   System Status: {data.get('system_status', 'N/A')}")
            print(f"   Recommended endpoint: {data.get('recommended_endpoint', 'N/A')}")
            
            if 'swasthyasetu_t5' in data:
                t5_info = data['swasthyasetu_t5']
                print(f"   SwasthyaSetu T5: {t5_info.get('status', 'N/A')}")
                print(f"   T5 Model loaded: {t5_info.get('is_loaded', 'N/A')}")
                print(f"   Device: {t5_info.get('device', 'N/A')}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Model info error: {e}")

if __name__ == "__main__":
    print("🏥 SwasthyaSetu Medical Summarizer - Braces Data Test")
    print("=" * 70)
    
    # Test model info first
    test_model_info()
    
    # Test SwasthyaSetu T5 endpoint
    test_swasthyasetu_t5_endpoint()
    
    # Test ultra-fast endpoint for comparison
    test_ultra_fast_endpoint()
    
    print("\n✨ Testing complete!")
    print("\n💡 Use /summarize/swasthyasetu-t5 for AI-powered summaries!")
    print("💡 Use /summarize/ultra-fast for instant results!")
